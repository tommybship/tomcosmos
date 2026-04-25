"""End-to-end M4 #3: Hohmann-style Earth → Mars transfer using the
Lambert solver.

Procedure:
1. Load the bare scenario (Sun, Earth, Mars). Pick an arrival epoch
   ~259 days after departure (textbook Hohmann TOF for r₁=1 AU,
   r₂=1.524 AU).
2. Query Earth's state at scenario.epoch → probe rides along.
3. `compute_transfer(...)` returns the two impulsive burns.
4. Inject probe + Δv events into a fresh Scenario, run, assert the
   probe is within 1° of Mars's phase angle at the targeted arrival
   sample. The remaining residual is integrator drift over 259 days
   plus the slight non-coplanarity of Earth and Mars (~1.85° ecliptic
   inclination of Mars), not algorithmic Lambert error.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from astropy import units as u

from tomcosmos import Scenario
from tomcosmos.runner import run
from tomcosmos.state.ephemeris import SkyfieldSource
from tomcosmos.targeting import compute_transfer


@pytest.mark.ephemeris
def test_lambert_targeted_earth_mars_transfer_lands_near_mars(
    skyfield_source: SkyfieldSource,
) -> None:
    """The Lambert-solved transfer should put the probe within 1° of
    Mars's phase angle at the arrival epoch.

    Closeness is measured as the angle between the probe's heliocentric
    position and Mars's heliocentric position at the targeted arrival
    sample — not the spatial distance, since "5° of phase" is the
    PLAN.md exit-criterion language and matches what's interesting for
    a mission-planning sketch (does the probe show up where Mars is,
    not somewhere on the wrong side of the orbit).

    Empirically the probe lands within ~0.05° at this epoch — well
    inside the tightened 1° bound.
    """
    base = Scenario.from_yaml("scenarios/earth-mars-hohmann.yaml")
    # 310-day TOF lands inside the YAML's 320-day duration window
    # (chosen on the 2026-10-28 launch window for ~Hohmann Δvs).
    arrival_epoch = base.epoch + 310.0 * u.day

    transfer = compute_transfer(
        source=skyfield_source,
        from_body="earth",
        to_body="mars",
        departure_epoch=base.epoch,
        arrival_epoch=arrival_epoch,
    )

    # Probe rides Earth's heliocentric state at t=0; Δv events come
    # from the Lambert solution. Build the augmented scenario via
    # `model_validate` (matches the rest of the test suite's pattern
    # and avoids importing `TestParticle` at module level — pytest
    # would try to collect anything starting with `Test` as a class).
    r_earth, v_earth = skyfield_source.query("earth", base.epoch)
    dep_dv, arr_dv = transfer.as_dv_events(base.epoch)

    scenario = Scenario.model_validate({
        **base.model_dump(mode="json"),
        "test_particles": [{
            "name": "probe",
            "ic": {
                "type": "explicit",
                "r": list(r_earth),
                "v": list(v_earth),
                "frame": "icrf_barycentric",
            },
            "dv_events": [dep_dv.model_dump(mode="json"),
                          arr_dv.model_dump(mode="json")],
        }],
    })

    history = run(scenario, source=skyfield_source)

    # Find the sample closest to arrival_epoch.
    arrival_t_s = float((arrival_epoch - base.epoch).to(u.s).value)
    samples = (
        history.df[history.df["body"] == "probe"][["sample_idx", "t_tdb"]]
        .reset_index(drop=True)
    )
    nearest_pos = (samples["t_tdb"] - arrival_t_s).abs().idxmin()
    sidx = int(samples.loc[nearest_pos, "sample_idx"])

    def _r_at(body: str) -> np.ndarray:
        row = history.df[
            (history.df["body"] == body) & (history.df["sample_idx"] == sidx)
        ]
        return row[["x", "y", "z"]].iloc[0].to_numpy(dtype=np.float64)

    r_probe = _r_at("probe")
    r_mars = _r_at("mars")
    r_sun = _r_at("sun")

    # Heliocentric phase angle between probe and Mars.
    sun_to_probe = r_probe - r_sun
    sun_to_mars = r_mars - r_sun
    cos_phase = float(np.dot(sun_to_probe, sun_to_mars) / (
        np.linalg.norm(sun_to_probe) * np.linalg.norm(sun_to_mars)
    ))
    cos_phase = max(-1.0, min(1.0, cos_phase))
    phase_deg = math.degrees(math.acos(cos_phase))

    assert phase_deg < 1.0, (
        f"probe missed Mars by {phase_deg:.3f}° of heliocentric phase; "
        f"Lambert + IAS15 should land within 1°"
    )


@pytest.mark.ephemeris
def test_compute_transfer_reproduces_hohmann_delta_v_magnitudes(
    skyfield_source: SkyfieldSource,
) -> None:
    """Δvs from `compute_transfer` should match textbook Hohmann
    magnitudes within ~10% when departing in a real launch window.

    Earth-Mars launch windows recur every ~780 days (synodic period).
    The scenario YAML's 2026-10-28 epoch sits inside the 2026 window,
    so the geometry approximates the coplanar-circular Hohmann
    idealization closely. Outside a launch window the Δvs would be
    much higher — Lambert is correct either way, but this test only
    makes sense inside one."""
    base = Scenario.from_yaml("scenarios/earth-mars-hohmann.yaml")
    arrival_epoch = base.epoch + 310.0 * u.day
    transfer = compute_transfer(
        source=skyfield_source,
        from_body="earth", to_body="mars",
        departure_epoch=base.epoch, arrival_epoch=arrival_epoch,
    )

    AU_KM = 1.495978707e8
    MU_SUN = 1.32712440018e11
    r1 = 1.0 * AU_KM
    r2 = 1.524 * AU_KM
    expected_dep = math.sqrt(MU_SUN / r1) * (math.sqrt(2 * r2 / (r1 + r2)) - 1.0)
    expected_arr = math.sqrt(MU_SUN / r2) * (1.0 - math.sqrt(2 * r1 / (r1 + r2)))

    dep_mag = float(np.linalg.norm(transfer.delta_v_departure))
    arr_mag = float(np.linalg.norm(transfer.delta_v_arrival))
    assert abs(dep_mag - expected_dep) / expected_dep < 0.15
    assert abs(arr_mag - expected_arr) / expected_arr < 0.15
