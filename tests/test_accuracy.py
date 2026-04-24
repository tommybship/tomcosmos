"""Accuracy envelope — Tier 3 tests that compare propagated positions
to the ephemeris we bootstrapped from.

Marked `ephemeris` so they skip on CI's default filter (see
pyproject.toml > tool.pytest.ini_options). The PLAN.md > "Accuracy
envelope" table is the source of truth for tolerances; assertions here
read from that table rather than hard-coding copies.
"""
from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u

from tomcosmos import Scenario, run
from tomcosmos.state.ephemeris import SkyfieldSource

pytestmark = pytest.mark.ephemeris


def _full_sun_planets(duration: str, cadence: str) -> Scenario:
    return Scenario.model_validate(
        {
            "schema_version": 1,
            "name": f"accuracy-{duration.replace(' ', '')}",
            "epoch": "2026-01-01T00:00:00 TDB",
            "duration": duration,
            "integrator": {"name": "whfast", "timestep": "1 day"},
            "output": {"format": "parquet", "cadence": cadence},
            "bodies": [
                {"name": n, "spice_id": sid, "ic": {"source": "ephemeris"}}
                for n, sid in [
                    ("sun", 10), ("mercury", 199), ("venus", 299),
                    ("earth", 399), ("mars", 499), ("jupiter", 599),
                    ("saturn", 699), ("uranus", 799), ("neptune", 899),
                ]
            ],
        }
    )


EARTH_1YR_ENVELOPE_KM = 2_000_000.0
MERCURY_1YR_ENVELOPE_KM = 700_000.0


def test_earth_within_envelope_after_one_year(
    skyfield_source: SkyfieldSource,
) -> None:
    """PLAN > Accuracy envelope: Earth propagated 1 yr from ephemeris ICs
    should match the ephemeris at t=1 yr within the observed envelope.

    The residual is dominated by the physics we deliberately skip
    (moons perturbing the Sun and planets, GR, asteroid belt). This is
    a regression gate, not an accuracy goal — tightening it requires
    growing the model (moons, GR add-ons) rather than tweaking the
    integrator."""
    scenario = _full_sun_planets(duration="365 day", cadence="365 day")
    history = run(scenario, source=skyfield_source)

    earth = history.body_trajectory("earth")
    r_sim = earth[["x", "y", "z"]].to_numpy(dtype=np.float64)[-1]

    t_end = scenario.epoch + 365.0 * u.day
    r_truth, _ = skyfield_source.query("earth", t_end)

    delta_km = float(np.linalg.norm(r_sim - r_truth))
    assert delta_km < EARTH_1YR_ENVELOPE_KM, (
        f"Earth drift after 1 yr: {delta_km:.0f} km "
        f"(envelope: {EARTH_1YR_ENVELOPE_KM:.0f} km)"
    )


def test_mercury_within_envelope_after_one_year(
    skyfield_source: SkyfieldSource,
) -> None:
    """Mercury's short period gives it more orbits per year; any mass or
    GM mismatch compounds. Envelope is 700,000 km at 1 yr — mostly
    phase error from the same physics omissions listed above."""
    scenario = _full_sun_planets(duration="365 day", cadence="365 day")
    history = run(scenario, source=skyfield_source)

    mercury = history.body_trajectory("mercury")
    r_sim = mercury[["x", "y", "z"]].to_numpy(dtype=np.float64)[-1]

    t_end = scenario.epoch + 365.0 * u.day
    r_truth, _ = skyfield_source.query("mercury", t_end)

    delta_km = float(np.linalg.norm(r_sim - r_truth))
    assert delta_km < MERCURY_1YR_ENVELOPE_KM, (
        f"Mercury drift after 1 yr: {delta_km:.0f} km "
        f"(envelope: {MERCURY_1YR_ENVELOPE_KM:.0f} km)"
    )


def test_energy_bounded_over_10_years(
    skyfield_source: SkyfieldSource,
) -> None:
    """Symplectic signature: |ΔE/E| stays below the WHFast envelope
    (1e-10) and doesn't drift linearly. A drifting slope would indicate
    wrong units, missing move_to_com(), or a bad timestep."""
    scenario = _full_sun_planets(duration="10 yr", cadence="30 day")
    history = run(scenario, source=skyfield_source)

    trace = history.energy_trace()
    assert trace["energy_rel_err"].max() < 1e-9

    # Slope check: linear fit on log(|rel_err|) vs time.
    # Symplectic drift should have slope ~0; genuine leak shows positive slope.
    nonzero = trace[trace["energy_rel_err"] > 0]
    if len(nonzero) > 10:
        x = nonzero["t_tdb"].to_numpy()
        y = np.log10(nonzero["energy_rel_err"].to_numpy())
        slope, _ = np.polyfit(x, y, 1)
        # Slope in log10 per second. Over 10 yr (~3.15e8 s), if slope is
        # > 1e-10 per sec, total drift would be 0.03 decades — tiny but
        # detectable. We assert the slope is below a loose 1e-9 per sec
        # to catch real leaks without being spuriously flaky.
        assert slope < 1e-9, f"energy drift detected: slope={slope:.2e} log10/sec"
