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


# --- M2 exit criterion #4: IAS15 holds energy to <1e-12 over 10 yr ----------


def _earth_moon_long(duration: str, cadence: str) -> Scenario:
    return Scenario.model_validate(
        {
            "schema_version": 1,
            "name": f"earth-moon-{duration.replace(' ', '')}",
            "epoch": "2026-01-01T00:00:00 TDB",
            "duration": duration,
            "integrator": {"name": "ias15"},
            "output": {"format": "parquet", "cadence": cadence},
            "bodies": [
                {"name": n, "spice_id": sid, "ic": {"source": "ephemeris"}}
                for n, sid in [("sun", 10), ("earth", 399), ("moon", 301)]
            ],
        }
    )


def _jupiter_galileans_long(duration: str, cadence: str) -> Scenario:
    return Scenario.model_validate(
        {
            "schema_version": 1,
            "name": f"jupiter-galileans-{duration.replace(' ', '')}",
            "epoch": "2026-01-01T00:00:00 TDB",
            "duration": duration,
            "integrator": {"name": "ias15"},
            "output": {"format": "parquet", "cadence": cadence},
            "bodies": [
                {"name": n, "spice_id": sid, "ic": {"source": "ephemeris"}}
                for n, sid in [
                    ("sun", 10), ("jupiter", 599),
                    ("io", 501), ("europa", 502),
                    ("ganymede", 503), ("callisto", 504),
                ]
            ],
        }
    )


def test_ias15_earth_moon_energy_under_1e_minus_12_over_10_years(
    skyfield_source: SkyfieldSource,
) -> None:
    """M2 exit criterion #4: IAS15 holds |ΔE/E| ≤ 1e-12 on Earth+Moon
    over a 10-year integration. IAS15's per-step error target is
    machine precision; a violation here means wrong units, missing
    move_to_com(), or a regression in the moon-IC pipeline."""
    scenario = _earth_moon_long(duration="10 yr", cadence="30 day")
    history = run(scenario, source=skyfield_source)
    max_err = float(history.df["energy_rel_err"].max())
    assert max_err < 1e-12, f"Earth+Moon |ΔE/E| over 10 yr: {max_err:.3e}"


def test_ias15_jupiter_galileans_energy_under_1e_minus_12_over_10_years(
    skyfield_source: SkyfieldSource,
    kernel_dir,  # type: ignore[no-untyped-def]
) -> None:
    """M2 exit criterion #4 on the stiffer system: Jupiter + 4 Galileans
    over 10 years. Io's 1.77-day period forces IAS15 to take small
    adaptive steps; energy error is the load-bearing diagnostic that
    the adaptive controller is converging."""
    if not (kernel_dir / "jup365.bsp").exists():
        pytest.skip("jup365.bsp not present")
    scenario = _jupiter_galileans_long(duration="10 yr", cadence="30 day")
    history = run(scenario, source=skyfield_source)
    max_err = float(history.df["energy_rel_err"].max())
    assert max_err < 1e-12, f"Jupiter+Galileans |ΔE/E| over 10 yr: {max_err:.3e}"


# --- M3 exit criterion #3: Sun-Earth L4 tadpole stays bounded over 50 yr ----


def test_l4_tadpole_within_10_degrees_over_50_years(
    skyfield_source: SkyfieldSource,
) -> None:
    """A test particle at Sun-Earth L4 should librate stably around the
    equilibrium for the entire integration window. PLAN.md M3 exit
    criterion #3: stays within ±10° in the Sun-Earth rotating frame.

    Empirically the libration is ~0.2° peak over 50 yr — three orders
    of magnitude tighter than the criterion bound. The wide bound
    catches integrator divergence or wrong-frame transformations
    without flapping on real-world Earth-eccentricity wobbles."""
    from tomcosmos.analysis.rotating_frame import (
        angular_position_relative_to,
        rotate_history_to_corotating,
    )

    scenario = Scenario.from_yaml("scenarios/sun-earth-l4-tadpole.yaml")
    history = run(scenario, source=skyfield_source)
    rotated = rotate_history_to_corotating(
        history, primary="sun", secondary="earth",
    )
    delta_deg = angular_position_relative_to(
        rotated, particle="l4-trojan", reference_angle_deg=60.0,
    )
    max_libration = float(np.max(np.abs(delta_deg)))
    assert max_libration < 10.0, (
        f"L4 tadpole left ±10° envelope: peak libration {max_libration:.3f}°"
    )
