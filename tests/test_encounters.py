"""Tests for analysis.encounters Hill-sphere detection.

Synthesizes histories with deliberately-known crossings so the math is
gated independent of the integrator. Two real-integration tests at the
end validate the end-to-end runner integration.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tomcosmos import Scenario, run
from tomcosmos.analysis.encounters import detect_hill_encounters
from tomcosmos.io.history import StateHistory
from tomcosmos.state.ephemeris import EphemerisSource


def _three_body_history(
    earth_r0_km: tuple[float, float, float] = (1.5e8, 0.0, 0.0),
    asteroid_offsets_km: list[tuple[float, float, float]] | None = None,
) -> StateHistory:
    """Build a synthetic three-body history (sun, earth, asteroid) with
    a fixed Earth and an asteroid whose position varies sample-by-sample
    via `asteroid_offsets_km` (relative to Earth). Lets us drive
    Hill-crossings deterministically without any integrator math."""
    if asteroid_offsets_km is None:
        # Default trajectory: starts at 3e6 km, sweeps to 0.5e6 km, back to
        # 3e6 km. Earth Hill radius is ~1.5e6 km, so we expect 1 enter + 1 exit.
        asteroid_offsets_km = [
            (3.0e6, 0.0, 0.0),
            (2.0e6, 0.0, 0.0),
            (0.5e6, 0.0, 0.0),
            (2.0e6, 0.0, 0.0),
            (3.0e6, 0.0, 0.0),
        ]
    n = len(asteroid_offsets_km)
    rows = []
    for i, off in enumerate(asteroid_offsets_km):
        rows.append({"sample_idx": i, "t_tdb": float(i) * 86400.0,
                     "body": "sun",
                     "x": 0.0, "y": 0.0, "z": 0.0,
                     "vx": 0.0, "vy": 0.0, "vz": 0.0,
                     "terminated": False, "energy_rel_err": 0.0})
        rows.append({"sample_idx": i, "t_tdb": float(i) * 86400.0,
                     "body": "earth",
                     "x": earth_r0_km[0], "y": earth_r0_km[1], "z": earth_r0_km[2],
                     "vx": 0.0, "vy": 30.0, "vz": 0.0,
                     "terminated": False, "energy_rel_err": 0.0})
        rows.append({"sample_idx": i, "t_tdb": float(i) * 86400.0,
                     "body": "asteroid",
                     "x": earth_r0_km[0] + off[0],
                     "y": earth_r0_km[1] + off[1],
                     "z": earth_r0_km[2] + off[2],
                     "vx": 0.0, "vy": 0.0, "vz": 0.0,
                     "terminated": False, "energy_rel_err": 0.0})
    df = pd.DataFrame(rows)

    scenario = Scenario.model_validate({
        "schema_version": 1, "name": "encounters-test",
        "epoch": "2026-01-01T00:00:00 TDB",
        "duration": f"{n} day",
        "integrator": {"name": "ias15"},
        "output": {"format": "parquet", "cadence": "1 day"},
        "bodies": [
            {"name": "sun",   "spice_id": 10,  "ic": {"source": "ephemeris"}},
            {"name": "earth", "spice_id": 399, "ic": {"source": "ephemeris"}},
        ],
        "test_particles": [
            {"name": "asteroid", "ic": {"type": "explicit",
                                        "r": list(earth_r0_km),
                                        "v": [0.0, 0.0, 0.0]}},
        ],
    })
    return StateHistory(
        df=df, scenario=scenario,
        body_names=("sun", "earth", "asteroid"),
    )


# --- Pure detector tests (no integration) -----------------------------------


def test_detects_one_enter_one_exit_for_simple_flyby() -> None:
    history = _three_body_history()
    events = detect_hill_encounters(history)
    assert len(events) == 2
    assert list(events["kind"]) == ["encounter_enter", "encounter_exit"]
    assert (events["particle"] == "asteroid").all()
    assert (events["body"] == "earth").all()


def test_event_includes_distance_and_hill_radius() -> None:
    history = _three_body_history()
    events = detect_hill_encounters(history)
    enter = events.iloc[0]
    # Enter event sample should be where the asteroid first dropped
    # below the Hill radius. The fixture's sample 2 is the closest
    # approach (0.5e6 km offset), well below Earth's ~1.5e6 km Hill.
    assert enter["sample_idx"] == 2
    assert float(enter["distance_km"]) < float(enter["hill_radius_km"])
    # Earth Hill radius approximately a*(m/3M_sun)^(1/3); a≈1 AU; m=Earth mass.
    # Expected ~1.5e6 km.
    assert 1.0e6 < float(enter["hill_radius_km"]) < 2.0e6


def test_no_events_when_particle_stays_outside() -> None:
    history = _three_body_history(asteroid_offsets_km=[
        (5.0e6, 0.0, 0.0), (5.0e6, 0.0, 0.0), (5.0e6, 0.0, 0.0),
    ])
    events = detect_hill_encounters(history)
    assert events.empty


def test_initial_inside_emits_enter_at_sample_0() -> None:
    """A particle starting inside the Hill sphere should fire an enter
    event at sample 0 — there's no 'last sample outside' to compare
    against, but it's still inside, so flag it."""
    history = _three_body_history(asteroid_offsets_km=[
        (0.5e6, 0.0, 0.0), (0.5e6, 0.0, 0.0), (5.0e6, 0.0, 0.0),
    ])
    events = detect_hill_encounters(history)
    enters = events[events["kind"] == "encounter_enter"]
    assert int(enters["sample_idx"].iloc[0]) == 0


def test_re_entry_emits_two_enters() -> None:
    """In, out, in → enter, exit, enter (three events)."""
    history = _three_body_history(asteroid_offsets_km=[
        (5.0e6, 0.0, 0.0),
        (0.5e6, 0.0, 0.0),  # inside
        (5.0e6, 0.0, 0.0),  # outside
        (0.5e6, 0.0, 0.0),  # inside again
    ])
    events = detect_hill_encounters(history)
    assert list(events["kind"]) == [
        "encounter_enter", "encounter_exit", "encounter_enter",
    ]


def test_no_test_particles_no_events() -> None:
    """With zero test particles, the detector returns an empty frame
    even if the run has multiple massive bodies."""
    history = _three_body_history()
    # Strip the test particle out of the scenario.
    scenario = Scenario.model_validate({
        **history.scenario.model_dump(mode="json"),
        "test_particles": [],
    })
    history_no_tp = StateHistory(
        df=history.df[history.df["body"] != "asteroid"].reset_index(drop=True),
        scenario=scenario,
        body_names=("sun", "earth"),
    )
    events = detect_hill_encounters(history_no_tp)
    assert events.empty


# --- End-to-end integration tests -------------------------------------------


@pytest.mark.ephemeris
def test_runner_attaches_events_to_history(ephemeris_source: EphemerisSource) -> None:
    """A direct flyby launched at Earth produces enter/exit events
    on the resulting StateHistory."""
    epoch_scenario = Scenario.model_validate({
        "schema_version": 1, "name": "flyby-runner",
        "epoch": "2026-04-23T00:00:00 TDB",
        "duration": "30 day",
        "integrator": {"name": "ias15"},
        "output": {"format": "parquet", "cadence": "1 day"},
        "bodies": [
            {"name": "sun",   "spice_id": 10,  "ic": {"source": "ephemeris"}},
            {"name": "earth", "spice_id": 399, "ic": {"source": "ephemeris"}},
        ],
        "test_particles": [{
            "name": "asteroid",
            "ic": {"type": "explicit", "r": [0.0, 0.0, 0.0], "v": [0.0, 0.0, 0.0]},
        }],
    })
    r_earth, v_earth = ephemeris_source.query("earth", epoch_scenario.epoch)
    r_init = (r_earth + np.array([5.0e6, 0.0, 0.0])).tolist()
    v_init = (v_earth + np.array([-10.0, 0.0, 0.0])).tolist()
    scenario = Scenario.model_validate({
        **epoch_scenario.model_dump(mode="json"),
        "test_particles": [{
            "name": "asteroid",
            "ic": {"type": "explicit", "r": r_init, "v": v_init},
        }],
    })

    history = run(scenario, source=ephemeris_source)
    assert history.events is not None
    assert len(history.events) >= 1
    enter = history.events[history.events["kind"] == "encounter_enter"]
    assert (enter["body"] == "earth").any()
    assert (enter["particle"] == "asteroid").any()


@pytest.mark.ephemeris
def test_l4_tadpole_has_no_encounters(ephemeris_source: EphemerisSource) -> None:
    """The Sun-Earth L4 trojan never approaches Earth's Hill sphere — its
    closest approach is ~1 AU away. If our detector spuriously fires here,
    the Hill-radius math is wrong."""
    scenario = Scenario.from_yaml("scenarios/sun-earth-l4-tadpole.yaml")
    history = run(scenario, source=ephemeris_source)
    assert history.events is not None
    assert history.events.empty, (
        f"L4 trojan should have no Hill encounters; got {len(history.events)} events"
    )


@pytest.mark.ephemeris
def test_mode_a_with_no_source_returns_empty_events(
    ephemeris_source: EphemerisSource,
) -> None:
    """Mode A scenarios have no `scenario.bodies` for the detector to
    pivot from; without an EphemerisSource the detector has nowhere
    to read planet positions back from. Return an empty event log
    rather than raising — the trajectory data in StateHistory is
    still authoritative; only the side-output is missing."""
    # Synthesize a tiny Mode A history without going through run() —
    # avoids needing the ASSIST kernels for this code-path test.
    df = pd.DataFrame([
        {"sample_idx": i, "t_tdb": i * 86400.0, "body": "p",
         "x": 1.5e8 + i*1e3, "y": 0.0, "z": 0.0,
         "vx": 0.0, "vy": 30.0, "vz": 0.0,
         "terminated": False, "energy_rel_err": 0.0}
        for i in range(5)
    ])
    scenario = Scenario.model_validate({
        "schema_version": 1, "name": "mode-a-no-source",
        "epoch": "2026-01-01T00:00:00 TDB", "duration": "5 day",
        "integrator": {"name": "ias15", "ephemeris_perturbers": True},
        "output": {"format": "parquet", "cadence": "1 day"},
        "test_particles": [{"name": "p", "ic": {
            "type": "explicit", "r": [1.5e8, 0.0, 0.0], "v": [0.0, 30.0, 0.0],
        }}],
    })
    history = StateHistory(df=df, scenario=scenario, body_names=("p",))
    events = detect_hill_encounters(history)  # no source kwarg
    assert events.empty
    # With source, but the synthetic test particle drifts away from
    # Earth steadily — no real encounters expected, but the call path
    # exercises the Mode A code without crashing.
    events_with_source = detect_hill_encounters(history, source=ephemeris_source)
    assert events_with_source.empty


@pytest.mark.ephemeris
def test_mode_a_detects_earth_flyby_against_real_ephemeris(
    ephemeris_source: EphemerisSource,
) -> None:
    """Synthesize a Mode A history where the test particle's position
    deliberately passes within Earth's Hill sphere over the course of
    the run. The detector reads Earth's ephemeris-driven positions
    back from skyfield and must flag the crossing.

    This is the headline Mode A capability: scenario.bodies is empty
    (ASSIST drives the major-body gravity at integration time), but
    NEO-vs-Earth encounter analysis still works because the analysis
    layer can re-query the ephemeris for post-hoc geometry."""
    from astropy import units as u
    from astropy.time import Time

    epoch = Time("2026-01-01T00:00:00", scale="tdb")
    n_samples = 11
    sample_times_s = np.arange(n_samples, dtype=np.float64) * 86400.0  # 1-day cadence
    times_abs = epoch + sample_times_s * u.s

    # Earth's actual barycentric trajectory over the window.
    earth_xyz, _ = ephemeris_source.query_many("earth", times_abs)

    # Particle trajectory: glued to Earth + a small offset that
    # collapses through zero in the middle of the window. At Earth's
    # ~1 AU semi-major axis the Hill radius is ~1.5e6 km, so an offset
    # sweeping from 3e6 → 0.5e6 → 3e6 km along +x produces exactly
    # one enter + one exit.
    offsets_km = np.array([3.0e6, 2.5e6, 2.0e6, 1.5e6, 1.0e6,
                           0.5e6, 1.0e6, 1.5e6, 2.0e6, 2.5e6, 3.0e6])
    particle_xyz = earth_xyz.copy()
    particle_xyz[:, 0] += offsets_km

    rows = []
    for i in range(n_samples):
        rows.append({
            "sample_idx": i, "t_tdb": sample_times_s[i], "body": "asteroid",
            "x": float(particle_xyz[i, 0]),
            "y": float(particle_xyz[i, 1]),
            "z": float(particle_xyz[i, 2]),
            "vx": 0.0, "vy": 0.0, "vz": 0.0,
            "terminated": False, "energy_rel_err": 0.0,
        })
    df = pd.DataFrame(rows)
    scenario = Scenario.model_validate({
        "schema_version": 1, "name": "mode-a-flyby",
        "epoch": f"{epoch.isot} TDB", "duration": "10 day",
        "integrator": {"name": "ias15", "ephemeris_perturbers": True},
        "output": {"format": "parquet", "cadence": "1 day"},
        "test_particles": [{"name": "asteroid", "ic": {
            "type": "explicit",
            "r": [float(particle_xyz[0, 0]), float(particle_xyz[0, 1]), float(particle_xyz[0, 2])],
            "v": [0.0, 0.0, 0.0],
        }}],
    })
    history = StateHistory(df=df, scenario=scenario, body_names=("asteroid",))

    events = detect_hill_encounters(history, source=ephemeris_source)
    assert not events.empty, "Earth flyby should have produced encounter events"
    earth_events = events[events["body"] == "earth"]
    assert len(earth_events) == 2, (
        f"expected 1 enter + 1 exit for earth, got: {earth_events.to_dict('records')}"
    )
    assert list(earth_events["kind"]) == ["encounter_enter", "encounter_exit"]
    assert (earth_events["particle"] == "asteroid").all()


def test_events_sidecar_round_trip(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Writing a history with events and reading it back preserves the
    event log via the sidecar parquet."""
    history = _three_body_history()
    events = detect_hill_encounters(history)
    history = StateHistory(
        df=history.df, scenario=history.scenario,
        body_names=history.body_names, events=events,
    )
    out = tmp_path / "run.parquet"
    history.to_parquet(out)
    sidecar = tmp_path / "run.events.parquet"
    assert sidecar.exists(), "events sidecar should be written when events present"

    reloaded = StateHistory.from_parquet(out)
    assert reloaded.events is not None
    assert len(reloaded.events) == len(events)
    assert list(reloaded.events["kind"]) == list(events["kind"])
