"""Top-level orchestration: Scenario → StateHistory.

`run()` is the library API the CLI's `tomcosmos run` command wraps.
Most of the heavy lifting lives in specialized modules (ephemeris
query, IC resolution, REBOUND wrapper); this file stitches them
together and owns the integration loop.

As of part 6, `run()` captures RunMetadata and (optionally) writes a
Parquet output. Output-path resolution follows PLAN.md > "Output paths":
  - If `scenario.output.path` is set, use it (relative paths resolve
    under `config.runs_dir()`).
  - Otherwise default to `runs/<scenario_name>__<utc_iso_basic>.parquet`
    so reruns of the same scenario don't silently overwrite.
"""
from __future__ import annotations

import math
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import rebound
from astropy import units as u

from tomcosmos.config import runs_dir
from tomcosmos.io.diagnostics import RunMetadata, capture_metadata
from tomcosmos.io.history import COLUMNS, StateHistory
from tomcosmos.state.ephemeris import EphemerisSource, SkyfieldSource
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
    e0 = sim.energy()

    start = datetime.now(UTC)
    records: list[dict[str, object]] = []
    for sample_idx, t_yr in enumerate(sample_times_yr):
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
    events = detect_hill_encounters(history)
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
