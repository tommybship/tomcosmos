"""Top-level orchestration: Scenario → StateHistory.

`run()` is the library API the CLI's `tomcosmos run` command wraps.
Most of the heavy lifting lives in specialized modules (ephemeris
query, IC resolution, REBOUND wrapper); this file stitches them
together and owns the integration loop.

The runner keeps time in **seconds since scenario.epoch** and
velocities in **km/s** at its boundaries (the StateHistory `t_tdb`
column, the burn-timeline tuples, the row-emission logic). The sim's
internal units (Mode B: AU / yr / Msun, Mode A: AU / day / Msun) are
treated as an implementation detail — `state.sim_units` translates
them at the I/O boundary so this loop is mode-agnostic.

As of M4, the integration loop interleaves Δv burns with output
samples: at each step, all burns due before the next sample are
applied (integrate to burn time, modify particle velocity, continue),
so an arbitrary number of impulses can land between any two samples
without losing precision. The applied burns are recorded in the
returned StateHistory's `events` log alongside Hill-sphere encounters.
"""
from __future__ import annotations

import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import rebound
from astropy import units as u

from tomcosmos.config import runs_dir
from tomcosmos.io.diagnostics import RunMetadata, capture_metadata
from tomcosmos.io.history import COLUMNS, StateHistory
from tomcosmos.state.ephemeris import EphemerisSource
from tomcosmos.state.events import DeltaVEvent
from tomcosmos.state.frames import ecliptic_to_icrf
from tomcosmos.state.ic import (
    ResolvedBody,
    ResolvedTestParticle,
    resolve_scenario,
)
from tomcosmos.state.integrator import build_simulation
from tomcosmos.state.scenario import Scenario
from tomcosmos.state.sim_units import (
    length_unit_to_km,
    time_unit_in_seconds,
    velocity_unit_to_kms,
)


def run(
    scenario: Scenario,
    source: EphemerisSource | None = None,
    *,
    write: bool = False,
    overwrite: bool = False,
    allow_dirty: bool = True,
) -> StateHistory:
    """Integrate a `Scenario` end-to-end and return an in-memory StateHistory.

    If `write=True`, also persists the result to Parquet at the path
    resolved from `scenario.output.path` (or a timestamped default).

    Parameters
    ----------
    scenario:
        Parsed and validated Scenario.
    source:
        Ephemeris source; defaults to an `EphemerisSource` rooted at the
        configured kernel directory and DE440s.
    write:
        Persist to Parquet on completion. Library default is False
        (stay in memory); the CLI passes True.
    overwrite:
        Allow overwriting an existing output path. Only consulted when
        `write=True`.
    allow_dirty:
        Allow running with uncommitted changes in the git working tree.
        Library default is True (user owns their env); CLI passes False
        so `tomcosmos run` enforces clean-tree discipline.
    """
    if source is None:
        source = EphemerisSource()
    source.require_covers(scenario.epoch, scenario.duration)

    bodies, particles = resolve_scenario(scenario, source)
    sim = build_simulation(
        bodies, particles, scenario.integrator, epoch=scenario.epoch,
    )

    # Cache once; sim.units doesn't change after build_simulation returns.
    time_unit_s = time_unit_in_seconds(sim)
    vel_to_kms = velocity_unit_to_kms(sim)
    pos_to_km = length_unit_to_km(sim)
    sim_t0 = float(sim.t)  # 0 in Mode B, days-past-J2000 in Mode A

    sample_offsets_s = _sample_grid_seconds(scenario)
    burn_timeline = _build_burn_timeline(scenario, time_unit_s, vel_to_kms)
    e0 = sim.energy()

    # Names in the order they were added — bodies first, then test particles —
    # which matches REBOUND's particle-array indexing. Both ASSIST and Mode B
    # paths preserve insertion order, so a single name list lets us skip
    # per-particle hash lookups inside the integration loop.
    all_names: list[str] = [b.name for b in bodies] + [p.name for p in particles]
    n_particles = len(all_names)
    n_samples = len(sample_offsets_s)
    n_rows = n_samples * n_particles

    # Pre-allocate every output column once, then fill via numpy slice
    # assignment inside the integration loop. The integration body does
    # zero per-particle Python work — `serialize_particle_data` is one
    # C-side copy of every particle's state; the assembly below is six
    # contiguous-stride slice writes plus four constant-fill writes.
    # For 1,000 bodies × 365 samples that's six 365k-element numpy
    # operations vs the previous 365,000 Python dict allocations.
    sample_idx_col = np.repeat(np.arange(n_samples, dtype=np.int64), n_particles)
    t_tdb_col = np.repeat(np.asarray(sample_offsets_s, dtype=np.float64), n_particles)
    body_col = np.tile(np.asarray(all_names, dtype=object), n_samples)
    x_col = np.empty(n_rows, dtype=np.float64)
    y_col = np.empty(n_rows, dtype=np.float64)
    z_col = np.empty(n_rows, dtype=np.float64)
    vx_col = np.empty(n_rows, dtype=np.float64)
    vy_col = np.empty(n_rows, dtype=np.float64)
    vz_col = np.empty(n_rows, dtype=np.float64)
    energy_col = np.empty(n_rows, dtype=np.float64)
    terminated_col = np.zeros(n_rows, dtype=bool)

    # Buffers REBOUND's serialize_particle_data writes into — flat 3*N
    # arrays per its contract.
    xyz_buf = np.empty(3 * n_particles, dtype=np.float64)
    vxvyvz_buf = np.empty(3 * n_particles, dtype=np.float64)

    start = datetime.now(UTC)
    dv_events: list[DeltaVEvent] = []
    burn_idx = 0
    for sample_idx, sample_offset_s in enumerate(sample_offsets_s):
        sample_sim_t = sim_t0 + sample_offset_s / time_unit_s

        # Apply any burns scheduled at or before this sample. Integrate
        # to the burn instant, modify the particle's velocity, continue.
        # Burns are discrete per-particle events — not vectorizable.
        while (
            burn_idx < len(burn_timeline)
            and burn_timeline[burn_idx][0] <= sample_offset_s
        ):
            t_burn_offset_s, name, dv_sim, dv_kms = burn_timeline[burn_idx]
            sim.integrate(sim_t0 + t_burn_offset_s / time_unit_s)
            p = sim.particles[rebound.hash(name)]
            p.vx += float(dv_sim[0])
            p.vy += float(dv_sim[1])
            p.vz += float(dv_sim[2])
            dv_events.append(DeltaVEvent(
                sample_idx=sample_idx,
                t_tdb=t_burn_offset_s,
                particle=name,
                dv_kms=(float(dv_kms[0]), float(dv_kms[1]), float(dv_kms[2])),
            ))
            burn_idx += 1

        sim.integrate(sample_sim_t)
        rel_err = abs((sim.energy() - e0) / e0) if e0 != 0 else 0.0

        sim.serialize_particle_data(xyz=xyz_buf, vxvyvz=vxvyvz_buf)
        row_start = sample_idx * n_particles
        row_end = row_start + n_particles
        # Strided views: xyz_buf[0::3] = x of every particle, etc.
        x_col[row_start:row_end]  = xyz_buf[0::3] * pos_to_km
        y_col[row_start:row_end]  = xyz_buf[1::3] * pos_to_km
        z_col[row_start:row_end]  = xyz_buf[2::3] * pos_to_km
        vx_col[row_start:row_end] = vxvyvz_buf[0::3] * vel_to_kms
        vy_col[row_start:row_end] = vxvyvz_buf[1::3] * vel_to_kms
        vz_col[row_start:row_end] = vxvyvz_buf[2::3] * vel_to_kms
        energy_col[row_start:row_end] = rel_err
    end = datetime.now(UTC)

    df = pd.DataFrame({
        "sample_idx": sample_idx_col,
        "t_tdb": t_tdb_col,
        "body": pd.array(body_col, dtype="string"),
        "x": x_col,
        "y": y_col,
        "z": z_col,
        "vx": vx_col,
        "vy": vy_col,
        "vz": vz_col,
        "terminated": terminated_col,
        "energy_rel_err": energy_col,
    }, columns=list(COLUMNS))

    metadata = capture_metadata(
        scenario, source, start, end, allow_dirty=allow_dirty
    )
    all_names = tuple(b.name for b in bodies) + tuple(p.name for p in particles)
    history = StateHistory(
        df=df, scenario=scenario, body_names=all_names, metadata=metadata
    )

    # Post-hoc Hill-sphere encounter detection. Cheap (O(n_samples × n_test
    # × n_massive) vector ops on already-resolved positions) and lets us
    # keep the integration loop unaware of analysis. Sub-cadence flybys are
    # missed by construction — output cadence sets the resolution.
    from tomcosmos.analysis.encounters import detect_hill_encounters
    encounter_events = detect_hill_encounters(history)
    events = _merge_event_logs(encounter_events, dv_events)
    history = StateHistory(
        df=df, scenario=scenario, body_names=all_names,
        metadata=metadata, events=events,
    )

    if write:
        path = resolve_output_path(scenario, metadata)
        history.to_parquet(path, overwrite=overwrite)

    return history


def resolve_output_path(scenario: Scenario, metadata: RunMetadata) -> Path:
    """Where should `run()` write when persisting this scenario?

    - `scenario.output.path` explicit → honored (relative resolved under
      `config.runs_dir()`).
    - Default → `<runs_dir>/<scenario_name>__<utc_iso_basic>.parquet`,
      with the timestamp from `metadata.start_wallclock` so the filename
      matches what's embedded in the file.
    """
    if scenario.output.path is not None:
        p = Path(scenario.output.path)
        return p if p.is_absolute() else runs_dir() / p
    ts = datetime.fromisoformat(metadata.start_wallclock).strftime("%Y%m%dT%H%M%SZ")
    return runs_dir() / f"{scenario.name}__{ts}.parquet"


def _build_burn_timeline(
    scenario: Scenario,
    time_unit_s: float,
    vel_to_kms: float,
) -> list[tuple[float, str, np.ndarray, np.ndarray]]:
    """Return `(t_offset_s, name, dv_sim_units, dv_kms)` tuples sorted by `t_offset_s`.

    Burns are pulled from every body and test particle in the scenario,
    transformed from their declared frame into ICRF, and converted from
    km/s into the sim's native velocity unit (AU/yr in Mode B,
    AU/day in Mode A). The original km/s vector is preserved in the
    returned tuple for the event log so we don't have to round-trip
    the conversion later.
    """
    del time_unit_s  # currently unused; the dv conversion goes via vel_to_kms
    burns: list[tuple[float, str, np.ndarray, np.ndarray]] = []
    for entity in (*scenario.bodies, *scenario.test_particles):
        for ev in entity.dv_events:
            t_offset_s = float(ev.t_offset.to(u.s).value)
            dv_kms = np.asarray(ev.dv, dtype=np.float64)
            dv_icrf = _dv_to_icrf(dv_kms, ev.frame)
            dv_sim = dv_icrf / vel_to_kms
            burns.append((t_offset_s, entity.name, dv_sim, dv_icrf))
    burns.sort(key=lambda b: b[0])
    return burns


def _dv_to_icrf(dv_kms: np.ndarray, frame: str) -> np.ndarray:
    """Δv vectors are pure rotations of frame; the heliocentric/barycentric
    distinction (origin shift) doesn't apply because Δv has no origin."""
    if frame in ("icrf_barycentric", "icrf_heliocentric"):
        return dv_kms
    if frame in ("ecliptic_j2000_barycentric", "ecliptic_j2000_heliocentric"):
        return ecliptic_to_icrf(dv_kms)
    raise ValueError(f"unsupported dv frame: {frame!r}")


def _merge_event_logs(
    encounters: pd.DataFrame, dv_events: list[DeltaVEvent],
) -> pd.DataFrame:
    """Combine post-hoc encounter detections with the runner's dv log.

    Returns a single events DataFrame sorted by (t_tdb, kind) so reads
    can scan chronologically. Empty inputs are tolerated; the schema
    is set by `_empty_events_df()` (called transitively from the
    encounter detector) regardless of which side is empty."""
    if not dv_events:
        return encounters
    dv_df = pd.DataFrame.from_records([ev.to_row() for ev in dv_events])
    merged = dv_df if encounters.empty else pd.concat(
        [encounters, dv_df], ignore_index=True,
    )
    return merged.sort_values(["t_tdb", "kind"]).reset_index(drop=True)


def _sample_grid_seconds(scenario: Scenario) -> list[float]:
    """Sample times as seconds-since-epoch, inclusive of 0 and the final
    whole-cadence step.

    If the duration isn't an integer multiple of the cadence, the last
    sample lands at `n * cadence`, not at `duration` — the scenario's
    tail is trimmed to the last full cadence.
    """
    duration_s = float(scenario.duration.to(u.s).value)
    cadence_s = float(scenario.output.cadence.to(u.s).value)
    if cadence_s <= 0:
        raise ValueError("output cadence must be positive")
    n_samples = int(math.floor(duration_s / cadence_s)) + 1
    return [i * cadence_s for i in range(n_samples)]


__all__ = ["run", "resolve_output_path", "StateHistory", "ResolvedBody", "ResolvedTestParticle"]
