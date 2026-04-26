"""Tests for impulsive Δv events (M4 exit criterion #1).

Two angles of attack:

- Schema validation (cheap, no integration): structure / out-of-window
  rejections / frame coverage.
- Integration round-trip: build a deep-space scenario where solar
  gravity is negligible over the test horizon, apply a known Δv at a
  sample boundary, assert the post-burn velocity matches IC + Δv to
  numerical precision and the energy delta matches m·v·dv + (1/2)m|dv|²
  (the M4 #1 exit criterion).
"""
from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from tomcosmos import DeltaV, Scenario, run

AU_KM = 1.495978707e8


def _deep_space_scenario(
    m_kg: float, v_init_kms: tuple[float, float, float],
    burn: dict | None,
) -> Scenario:
    """A probe far enough from the Sun that solar gravity is negligible
    over a 30-day window — turns the integration into ballistic motion
    so we can test Δv applier independent of the dynamics.

    100 AU is past Pluto; sun gravity there is ~6e-12 m/s². Over 30 days
    that integrates to ~1.5e-5 km/s of acceleration error, well below
    the test tolerance."""
    scen = {
        "schema_version": 1, "name": "dv-test",
        "epoch": "2026-01-01T00:00:00 TDB",
        "duration": "30 day",
        "integrator": {"name": "ias15"},
        "output": {"format": "parquet", "cadence": "1 day"},
        "bodies": [
            {"name": "sun", "mass_kg": 1.989e30, "radius_km": 695700.0,
             "ic": {"source": "explicit", "r": [0, 0, 0], "v": [0, 0, 0]}},
            {"name": "probe", "mass_kg": m_kg, "radius_km": 1.0,
             "ic": {"source": "explicit",
                    "r": [100.0 * AU_KM, 0, 0],
                    "v": list(v_init_kms)}},
        ],
    }
    if burn is not None:
        scen["bodies"][1]["dv_events"] = [burn]
    return Scenario.model_validate(scen)


# --- Schema validation ------------------------------------------------------


def test_dv_events_default_to_empty_list() -> None:
    scenario = _deep_space_scenario(1.0e3, (0, 0, 0), burn=None)
    assert scenario.bodies[1].dv_events == []


def test_dv_event_at_t_zero_rejected() -> None:
    """t=0 burns are rejected by the Duration parser itself (it requires
    a positive value), so this gate fires before the dv-window validator."""
    with pytest.raises(ValidationError, match=r"duration must be positive"):
        _deep_space_scenario(1.0e3, (0, 0, 0),
                             burn={"t_offset": "0 day", "dv": [1, 0, 0]})


def test_dv_event_past_duration_rejected() -> None:
    """Burns scheduled at or after the scenario duration would never fire,
    so the dv-window model_validator rejects them."""
    with pytest.raises(ValidationError, match=r"dv event t_offset.*must be in"):
        _deep_space_scenario(1.0e3, (0, 0, 0),
                             burn={"t_offset": "60 day", "dv": [1, 0, 0]})


def test_dv_frame_default_is_icrf_barycentric() -> None:
    scenario = _deep_space_scenario(1.0e3, (0, 0, 0),
                                    burn={"t_offset": "10 day", "dv": [1, 0, 0]})
    assert scenario.bodies[1].dv_events[0].frame == "icrf_barycentric"


def test_dv_event_can_be_constructed_directly() -> None:
    """Pydantic instantiation as well as YAML — useful for in-test scenarios."""
    ev = DeltaV.model_validate({"t_offset": "5 day", "dv": [1.0, 2.0, 3.0]})
    assert ev.dv == (1.0, 2.0, 3.0)


# --- Integration: applier correctness --------------------------------------


def test_burn_applied_at_sample_boundary_matches_ic_plus_dv() -> None:
    """Burn fires at exactly t=10 day = sample 10. Probe is at rest in
    deep space, so the post-burn velocity at sample 10 should equal the
    applied Δv to high precision (modulo the negligible 100-AU solar
    gravity over 10 days)."""
    dv = (5.0, 0.0, 0.0)
    scenario = _deep_space_scenario(
        m_kg=1.0e3, v_init_kms=(0, 0, 0),
        burn={"t_offset": "10 day", "dv": list(dv)},
    )
    history = run(scenario)
    probe = history.df[
        (history.df["body"] == "probe") & (history.df["sample_idx"] == 10)
    ]
    v_after = probe[["vx", "vy", "vz"]].iloc[0].to_numpy()
    assert np.allclose(v_after, dv, atol=1e-3)


def test_burn_event_appears_in_event_log() -> None:
    dv = (5.0, 0.0, 0.0)
    scenario = _deep_space_scenario(
        m_kg=1.0e3, v_init_kms=(0, 0, 0),
        burn={"t_offset": "10 day", "dv": list(dv)},
    )
    history = run(scenario)
    assert history.events is not None
    delta_v_rows = history.events[history.events["kind"] == "delta_v"]
    assert len(delta_v_rows) == 1
    row = delta_v_rows.iloc[0]
    assert row["particle"] == "probe"
    assert float(row["dv_x_kms"]) == 5.0
    assert float(row["dv_y_kms"]) == 0.0
    assert float(row["dv_z_kms"]) == 0.0
    # t_tdb should equal the configured t_offset in seconds.
    assert abs(float(row["t_tdb"]) - 10 * 86400.0) < 1e-6


def test_energy_delta_matches_kinetic_formula() -> None:
    """M4 exit criterion #1: energy change matches m·|dv|²/2 for a
    burn applied to a body at rest. Compare with-burn vs no-burn
    histories at the burn sample boundary."""
    m_kg = 1.0e3
    dv = (3.0, 4.0, 0.0)  # |dv| = 5 km/s
    no_burn = run(_deep_space_scenario(m_kg, (0, 0, 0), burn=None))
    with_burn = run(_deep_space_scenario(
        m_kg, (0, 0, 0),
        burn={"t_offset": "10 day", "dv": list(dv)},
    ))

    def _v_at(history, sample: int) -> np.ndarray:  # type: ignore[no-untyped-def]
        row = history.df[
            (history.df["body"] == "probe") & (history.df["sample_idx"] == sample)
        ]
        return row[["vx", "vy", "vz"]].iloc[0].to_numpy(dtype=np.float64)

    v_pre = _v_at(no_burn, 10)
    v_post = _v_at(with_burn, 10)

    # KE in joules: m_kg · |v_kms|² / 2 · 1e6 (km²/s² → m²/s²).
    KE_factor = m_kg * 1e6 / 2.0
    delta_E_J = KE_factor * (float(np.dot(v_post, v_post))
                             - float(np.dot(v_pre, v_pre)))

    # Expected: ΔE = m·v_pre·dv + (1/2)·m·|dv|² (instantaneous, before
    # gravity does any work). At 100 AU and 10 days, v_pre is essentially
    # zero, so the dot-product term vanishes and ΔE ≈ (1/2)·m·|dv|².
    dv_arr = np.asarray(dv, dtype=np.float64)
    expected_J = (
        m_kg * 1e6 * float(np.dot(v_pre, dv_arr))
        + 0.5 * m_kg * 1e6 * float(np.dot(dv_arr, dv_arr))
    )
    rel_err = abs(delta_E_J - expected_J) / expected_J
    assert rel_err < 1e-4, (
        f"ΔE mismatch: got {delta_E_J:.3e} J, expected {expected_J:.3e} J, "
        f"relative error {rel_err:.3e}"
    )


def test_multiple_burns_sorted_and_applied_in_order() -> None:
    """Two burns at different times produce a velocity that's the sum
    of the two Δvs (deep space, so no orbital reshuffling between them)."""
    dv1 = (1.0, 0.0, 0.0)
    dv2 = (0.0, 2.0, 0.0)
    scen_dict = {
        "schema_version": 1, "name": "dv-multi",
        "epoch": "2026-01-01T00:00:00 TDB",
        "duration": "30 day",
        "integrator": {"name": "ias15"},
        "output": {"format": "parquet", "cadence": "1 day"},
        "bodies": [
            {"name": "sun", "mass_kg": 1.989e30, "radius_km": 695700.0,
             "ic": {"source": "explicit", "r": [0, 0, 0], "v": [0, 0, 0]}},
            {"name": "probe", "mass_kg": 1.0e3, "radius_km": 1.0,
             "ic": {"source": "explicit",
                    "r": [100.0 * AU_KM, 0, 0], "v": [0, 0, 0]},
             "dv_events": [
                 # Declared out of order on purpose — the runner must sort.
                 {"t_offset": "20 day", "dv": list(dv2)},
                 {"t_offset": "5 day",  "dv": list(dv1)},
             ]},
        ],
    }
    history = run(Scenario.model_validate(scen_dict))
    # Sample at the boundary right after the second burn (sample 20).
    # Comparing at the burn boundary keeps the test independent of the
    # accumulated gravitational drift over the rest of the window.
    probe = history.df[
        (history.df["body"] == "probe") & (history.df["sample_idx"] == 20)
    ]
    v_after_both = probe[["vx", "vy", "vz"]].iloc[0].to_numpy()
    expected = np.array(dv1) + np.array(dv2)
    assert np.allclose(v_after_both, expected, atol=5e-3)
    # Two delta_v rows in the event log, in time order.
    delta_v_rows = history.events[history.events["kind"] == "delta_v"]
    assert len(delta_v_rows) == 2
    times = delta_v_rows["t_tdb"].to_numpy()
    assert times[0] < times[1]


def test_dv_event_in_ecliptic_frame_rotates_correctly() -> None:
    """A pure +Z burn in the ecliptic frame should land mostly along
    +Y_icrf and +Z_icrf (rotated by the obliquity ε≈23.44°).

    Pure +Z_ecliptic = (0, -sin ε, +cos ε)_icrf. For a 1 km/s burn,
    the y-component is ~-0.397 and z-component ~+0.918 km/s."""
    scenario = _deep_space_scenario(
        m_kg=1.0e3, v_init_kms=(0, 0, 0),
        burn={"t_offset": "10 day", "dv": [0.0, 0.0, 1.0],
              "frame": "ecliptic_j2000_barycentric"},
    )
    history = run(scenario)
    probe = history.df[
        (history.df["body"] == "probe") & (history.df["sample_idx"] == 10)
    ]
    v = probe[["vx", "vy", "vz"]].iloc[0].to_numpy()
    assert abs(v[0]) < 1e-3
    assert abs(v[1] - (-np.sin(np.deg2rad(23.4392911)))) < 1e-3
    assert abs(v[2] - np.cos(np.deg2rad(23.4392911))) < 1e-3


def test_event_log_merges_dv_with_encounters(ephemeris_source) -> None:  # type: ignore[no-untyped-def]
    """A scenario that triggers BOTH an Earth-Hill encounter and a Δv
    burn produces both kinds in the same event log, sorted by time.

    Probe IC aimed at Earth's Hill sphere (encounter expected ~day 5-8);
    a tiny tangential burn fires at day 20, well after the encounter, so
    its perturbation doesn't change whether the encounter occurs."""
    r_earth, v_earth = ephemeris_source.query(
        "earth", __import__("astropy.time", fromlist=["Time"]).Time(
            "2026-04-23T00:00:00", scale="tdb"))
    r_init = (r_earth + np.array([5.0e6, 0.0, 0.0])).tolist()
    v_init = (v_earth + np.array([-10.0, 0.0, 0.0])).tolist()
    # Probe is a TestParticle (not a Body) so it doesn't perturb Earth's
    # trajectory — the encounter geometry stays the same as the M3
    # `test_runner_attaches_events_to_history` test that's already
    # exercised this flyby.
    scenario = Scenario.model_validate({
        "schema_version": 1, "name": "dv-and-encounter",
        "epoch": "2026-04-23T00:00:00 TDB",
        "duration": "30 day",
        "integrator": {"name": "ias15"},
        "output": {"format": "parquet", "cadence": "1 day"},
        "bodies": [
            {"name": "sun",   "spice_id": 10,  "ic": {"source": "ephemeris"}},
            {"name": "earth", "spice_id": 399, "ic": {"source": "ephemeris"}},
        ],
        "test_particles": [{
            "name": "probe",
            "ic": {"type": "explicit", "r": r_init, "v": v_init},
            "dv_events": [{"t_offset": "20 day", "dv": [0.001, 0.0, 0.0]}],
        }],
    })
    history = run(scenario, source=ephemeris_source)
    kinds = set(history.events["kind"].unique())
    assert "delta_v" in kinds
    assert "encounter_enter" in kinds
    # Sorted by t_tdb ascending.
    times = history.events["t_tdb"].to_numpy()
    assert all(times[i] <= times[i + 1] for i in range(len(times) - 1))


# --- Marker for the integration tests above to be skippable on CI without
# kernels: the only one that actually queries the ephemeris is the
# event-log merge test.
pytestmark = []  # noqa: PLR2004
test_event_log_merges_dv_with_encounters = pytest.mark.ephemeris(  # type: ignore[assignment]
    test_event_log_merges_dv_with_encounters
)
