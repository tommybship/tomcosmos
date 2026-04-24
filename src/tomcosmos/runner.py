"""Top-level orchestration: Scenario → StateHistory.

`run()` is the library API the CLI's `tomcosmos run` command wraps.
Most of the heavy lifting lives in specialized modules (ephemeris
query, IC resolution, REBOUND wrapper); this file stitches them
together and owns the integration loop.

Part 5 produces an in-memory StateHistory; Parquet persistence,
checkpointing, and structured diagnostics capture land in part 6.
"""
from __future__ import annotations

import math

import pandas as pd
import rebound
from astropy import units as u

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
) -> StateHistory:
    """Integrate a `Scenario` end-to-end and return an in-memory StateHistory.

    Steps:
      1. Resolve ephemeris source (skyfield by default; pass any
         EphemerisSource to override).
      2. Verify the scenario window is inside the source's coverage.
      3. Resolve every Body / TestParticle to ICRF-barycentric state.
      4. Build the REBOUND simulation (AU/yr/Msun internally).
      5. Sample at the scenario's output cadence, converting each row
         back to km / km·s⁻¹ / seconds at the I/O boundary.
      6. Record |ΔE/E₀| per sample and return.

    No Parquet I/O, no metadata capture yet — this is the minimal
    library-level integration loop. Part 6 wraps it with persistence.
    """
    if source is None:
        source = SkyfieldSource()
    source.require_covers(scenario.epoch, scenario.duration)

    bodies, particles = resolve_scenario(scenario, source)
    sim = build_simulation(bodies, particles, scenario.integrator)

    sample_times_yr = _sample_grid(scenario)
    e0 = sim.energy()

    records: list[dict[str, object]] = []
    for sample_idx, t_yr in enumerate(sample_times_yr):
        sim.integrate(t_yr)
        rel_err = abs((sim.energy() - e0) / e0) if e0 != 0 else 0.0
        t_s = t_yr * _YR_S
        for body in bodies:
            records.append(_row(sample_idx, t_s, body.name, sim, rel_err))
        for particle in particles:
            records.append(_row(sample_idx, t_s, particle.name, sim, rel_err))

    df = pd.DataFrame.from_records(records, columns=list(COLUMNS))
    # Narrow types after construction; from_records otherwise infers object.
    df = df.astype(
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

    all_names = tuple(b.name for b in bodies) + tuple(p.name for p in particles)
    return StateHistory(df=df, scenario=scenario, body_names=all_names)


def _sample_grid(scenario: Scenario) -> list[float]:
    """Sample times in years, inclusive of 0 and the final whole-cadence step.

    If the duration isn't an integer multiple of the cadence, the last sample
    lands at `n * cadence`, not at `duration` — which means the scenario's
    tail is trimmed to the last full cadence. We'll add an
    end-of-window-if-needed sample in a later part once downstream callers
    start caring about exact coverage.
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


# Keep the resolved-body types re-exportable without callers poking into
# state.ic, in case we want to expose them from `tomcosmos` later.
__all__ = ["run", "StateHistory", "ResolvedBody", "ResolvedTestParticle"]
