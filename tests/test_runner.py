"""Tests for the integration loop (`tomcosmos.run`) and StateHistory.

Most tests use an in-memory explicit-IC Sun-Earth scenario — no ephemeris
dependency, completes in milliseconds. The full sun+planets case is
ephemeris-marked so it skips when the kernel isn't available.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tomcosmos import Scenario, StateHistory, run
from tomcosmos.state.ephemeris import EphemerisSource

AU_KM = 1.495978707e8
EARTH_SPEED_KMS = 29.7847  # circular-orbit velocity at 1 AU


def _explicit_sun_earth_scenario(duration: str = "1 yr", cadence: str = "30 day") -> Scenario:
    """Build a minimal Sun-Earth scenario with explicit ICs at the scenario
    epoch — avoids needing an ephemeris during fast unit tests."""
    return Scenario.model_validate(
        {
            "schema_version": 1,
            "name": "two-body-test",
            "epoch": "2026-01-01T00:00:00 TDB",
            "duration": duration,
            "integrator": {"name": "whfast", "timestep": "1 day"},
            "output": {"format": "parquet", "cadence": cadence},
            "bodies": [
                {
                    "name": "sun",
                    "mass_kg": 1.989e30,
                    "radius_km": 695700.0,
                    "ic": {
                        "source": "explicit",
                        "r": [0.0, 0.0, 0.0],
                        "v": [0.0, 0.0, 0.0],
                    },
                },
                {
                    "name": "earth",
                    "mass_kg": 5.9724e24,
                    "radius_km": 6371.0,
                    "ic": {
                        "source": "explicit",
                        "r": [AU_KM, 0.0, 0.0],
                        "v": [0.0, EARTH_SPEED_KMS, 0.0],
                    },
                },
            ],
        }
    )


# A stub ephemeris so scenarios with no `ephemeris` ICs don't need a kernel.
class _NoEphemerisNeeded(EphemerisSource):  # type: ignore[misc]
    def __init__(self) -> None:  # skip the kernel load; nothing calls query
        pass

    def query(self, body, epoch):  # type: ignore[no-untyped-def]
        raise AssertionError("this scenario shouldn't need ephemeris queries")

    def available_bodies(self):  # type: ignore[override]
        return ()

    def time_range(self):  # type: ignore[override]
        from astropy.time import Time
        return (
            Time("1900-01-01", scale="tdb"),
            Time("2100-01-01", scale="tdb"),
        )


# --- Basic shape -------------------------------------------------------------


def test_run_returns_state_history() -> None:
    history = run(_explicit_sun_earth_scenario(), source=_NoEphemerisNeeded())
    assert isinstance(history, StateHistory)


def test_row_count_equals_samples_times_bodies() -> None:
    scenario = _explicit_sun_earth_scenario(duration="1 yr", cadence="30 day")
    history = run(scenario, source=_NoEphemerisNeeded())
    # 1 yr / 30 days + 1 (inclusive of t=0) = floor(12.175) + 1 = 13
    assert history.n_samples == 13
    assert len(history.df) == 13 * 2


def test_first_sample_matches_input_ics_after_com_shift() -> None:
    """Sample_idx=0 is taken AFTER move_to_com(), so positions are shifted by
    the Sun's fraction of (Earth + Sun) relative to the original pair. That
    shift is mass_earth / (mass_earth + mass_sun) ≈ 3e-6 of 1 AU."""
    history = run(_explicit_sun_earth_scenario(), source=_NoEphemerisNeeded())
    earth_traj = history.body_trajectory("earth")
    # Earth.x at t=0 should be ~1 AU minus the ~3e-6 AU COM shift.
    assert abs(earth_traj.loc[0, "x"] - AU_KM) < 1e3  # within 1000 km of 1 AU


def test_energy_trace_is_per_sample_not_per_row() -> None:
    history = run(_explicit_sun_earth_scenario(), source=_NoEphemerisNeeded())
    trace = history.energy_trace()
    assert len(trace) == history.n_samples
    # First sample: E0 itself, so rel err is 0.
    assert trace.loc[0, "energy_rel_err"] == pytest.approx(0.0, abs=1e-12)


def test_energy_stays_bounded_over_one_year() -> None:
    history = run(_explicit_sun_earth_scenario(), source=_NoEphemerisNeeded())
    trace = history.energy_trace()
    assert trace["energy_rel_err"].max() < 1e-6


def test_earth_returns_near_start_after_one_year() -> None:
    history = run(_explicit_sun_earth_scenario(duration="1 yr"), source=_NoEphemerisNeeded())
    earth = history.body_trajectory("earth")
    first = earth.iloc[0]
    last = earth.iloc[-1]
    dr_km = math.sqrt(
        (last["x"] - first["x"]) ** 2
        + (last["y"] - first["y"]) ** 2
        + (last["z"] - first["z"]) ** 2
    )
    # Earth should be within a few million km of its starting point after
    # one Kepler period. Wider tolerance because last sample may be up to
    # one cadence short of a full period.
    assert dr_km < 5e7  # 0.3 AU


def test_time_column_is_seconds_from_epoch() -> None:
    history = run(_explicit_sun_earth_scenario(), source=_NoEphemerisNeeded())
    trace = history.energy_trace()
    # 30-day cadence means t_tdb increments by 30 * 86400 = 2,592,000 s.
    diffs = trace["t_tdb"].diff().dropna()
    assert np.allclose(diffs, 30 * 86400.0, rtol=1e-10)


# --- StateHistory accessors --------------------------------------------------


def test_body_trajectory_returns_sorted_by_time() -> None:
    history = run(_explicit_sun_earth_scenario(), source=_NoEphemerisNeeded())
    earth = history.body_trajectory("earth")
    assert list(earth.columns) == ["t_tdb", "x", "y", "z", "vx", "vy", "vz"]
    assert earth["t_tdb"].is_monotonic_increasing


def test_body_trajectory_unknown_body_raises() -> None:
    history = run(_explicit_sun_earth_scenario(), source=_NoEphemerisNeeded())
    with pytest.raises(KeyError, match="pluto"):
        history.body_trajectory("pluto")


def test_body_names_populated() -> None:
    history = run(_explicit_sun_earth_scenario(), source=_NoEphemerisNeeded())
    assert set(history.body_names) == {"sun", "earth"}


# --- Full scenario (ephemeris-backed) ---------------------------------------


@pytest.mark.ephemeris
def test_sun_planets_integrates_and_returns_history(
    ephemeris_source: EphemerisSource,
) -> None:
    scenario = Scenario.model_validate(
        {
            "schema_version": 1,
            "name": "sun-planets-short",
            "epoch": "2026-04-23T00:00:00 TDB",
            "duration": "30 day",  # keep short for CI friendliness
            "integrator": {"name": "whfast", "timestep": "1 day"},
            "output": {"format": "parquet", "cadence": "5 day"},
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
    history = run(scenario, source=ephemeris_source)
    assert history.n_samples == 7  # floor(30/5) + 1
    assert len(history.df) == 7 * 9
    assert history.energy_trace()["energy_rel_err"].max() < 1e-8
    # Every body should appear in every sample.
    counts = history.df.groupby("body").size()
    assert (counts == 7).all()
