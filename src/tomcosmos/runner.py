"""Top-level orchestration: Scenario → StateHistory.

`run()` is the library API the CLI's `tomcosmos run` command wraps.
Most of the heavy lifting lives in specialized modules (ephemeris
query, IC resolution, REBOUND wrapper); this file stitches them
together and owns the integration loop.

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
from tomcosmos.state.ephemeris import EphemerisSource, SkyfieldSource
from tomcosmos.state.events import DeltaVEvent
from tomcosmos.state.frames import ecliptic_to_icrf
from tomcosmos.state.ic import (
    ResolvedBody,
    ResolvedTestParticle,
    resolve_scenario,
)
from tomcosmos.state.integrator import build_simulation
from tomcosmos.state.scenario import Scenario

# Conversion factors out of REBOUND's AU/yr unit system back to our I/O
# boundary (km, km/s, s). Values match astropy's u.AU and u.yr (IAU 2012
# and Julian year respectively).
_AU_KM: float = float((1.0 * u.AU).to(u.km).value)
_YR_S: float = float((1.0 * u.yr).to(u.s).value)
_AU_PER_YR_TO_KM_PER_S: float = _AU_KM / _YR_S


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
        Ephemeris backend; defaults to a SkyfieldSource using DE440s.
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
        source = SkyfieldSource()
    source.require_covers(scenario.epoch, scenario.duration)

    bodies, particles = resolve_scenario(scenario, source)
    sim = build_simulation(bodies, particles, scenario.integrator)

    sample_times_yr = _sample_grid(scenario)
    burn_timeline = _build_burn_timeline(scenario)
    e0 = sim.energy()

    start = datetime.now(UTC)
    records: list[dict[str, object]] = []
    dv_events: list[DeltaVEvent] = []
    burn_idx = 0
    for sample_idx, t_yr in enumerate(sample_times_yr):
        # Apply any burns scheduled at or before this sample time. We
        # integrate to the burn time, modify the particle's velocity,
        # then continue. This preserves the discontinuous-impulse model
        # exactly across whfast / ias15 / mercurius.
        while burn_idx < len(burn_timeline) and burn_timeline[burn_idx][0] <= t_yr:
            t_burn_yr, name, dv_au_yr, dv_kms = burn_timeline[burn_idx]
            sim.integrate(t_burn_yr)
            p = sim.particles[rebound.hash(name)]
            p.vx += float(dv_au_yr[0])
            p.vy += float(dv_au_yr[1])
            p.vz += float(dv_au_yr[2])
            dv_events.append(DeltaVEvent(
                sample_idx=sample_idx,
                t_tdb=t_burn_yr * _YR_S,
                particle=name,
                dv_kms=(float(dv_kms[0]), float(dv_kms[1]), float(dv_kms[2])),
            ))
            burn_idx += 1

        sim.integrate(t_yr)
        rel_err = abs((sim.energy() - e0) / e0) if e0 != 0 else 0.0
        t_s = t_yr * _YR_S
        for body in bodies:
            records.append(_row(sample_idx, t_s, body.name, sim, rel_err))
        for particle in particles:
            records.append(_row(sample_idx, t_s, particle.name, sim, rel_err))
    end = datetime.now(UTC)

    df = pd.DataFrame.from_records(records, columns=list(COLUMNS)).astype(
        {
            "sample_idx": "int64",
            "t_tdb": "float64",
            "body": "string",
            "x": "float64",
            "y": "float64",
            "z": "float64",
            "vx": "float64",
            "vy": "float64",
            "vz": "float64",
            "terminated": "bool",
            "energy_rel_err": "float64",
        }
    )

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
) -> list[tuple[float, str, np.ndarray, np.ndarray]]:
    """Return `(t_yr, name, dv_au_per_yr, dv_kms)` tuples sorted by `t_yr`.

    Burns are pulled from every body and test particle in the scenario,
    transformed from their declared frame into ICRF, and converted from
    km/s into REBOUND's AU/yr units in one pass. The original km/s
    vector is preserved in the returned tuple for the event log so we
    don't have to round-trip the conversion later.
    """
    burns: list[tuple[float, str, np.ndarray, np.ndarray]] = []
    for entity in (*scenario.bodies, *scenario.test_particles):
        for ev in entity.dv_events:
            t_yr = float(ev.t_offset.to(u.yr).value)
            dv_kms = np.asarray(ev.dv, dtype=np.float64)
            dv_icrf = _dv_to_icrf(dv_kms, ev.frame)
            dv_au_yr = dv_icrf / _AU_PER_YR_TO_KM_PER_S
            burns.append((t_yr, entity.name, dv_au_yr, dv_icrf))
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


def _sample_grid(scenario: Scenario) -> list[float]:
    """Sample times in years, inclusive of 0 and the final whole-cadence step.

    If the duration isn't an integer multiple of the cadence, the last sample
    lands at `n * cadence`, not at `duration` — the scenario's tail is
    trimmed to the last full cadence.
    """
    duration_yr = float(scenario.duration.to(u.yr).value)
    cadence_yr = float(scenario.output.cadence.to(u.yr).value)
    if cadence_yr <= 0:
        raise ValueError("output cadence must be positive")
    n_samples = int(math.floor(duration_yr / cadence_yr)) + 1
    return [i * cadence_yr for i in range(n_samples)]


def _row(
    sample_idx: int,
    t_s: float,
    name: str,
    sim: rebound.Simulation,
    energy_rel_err: float,
) -> dict[str, object]:
    p = sim.particles[rebound.hash(name)]
    return {
        "sample_idx": sample_idx,
        "t_tdb": t_s,
        "body": name,
        "x": p.x * _AU_KM,
        "y": p.y * _AU_KM,
        "z": p.z * _AU_KM,
        "vx": p.vx * _AU_PER_YR_TO_KM_PER_S,
        "vy": p.vy * _AU_PER_YR_TO_KM_PER_S,
        "vz": p.vz * _AU_PER_YR_TO_KM_PER_S,
        "terminated": False,
        "energy_rel_err": energy_rel_err,
    }


__all__ = ["run", "resolve_output_path", "StateHistory", "ResolvedBody", "ResolvedTestParticle"]
