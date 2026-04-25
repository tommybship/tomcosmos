"""Tests for analysis.rotating_frame.

Synthesizes circular two-body trajectories so the assertions don't
depend on the integrator or ephemeris — what we're testing here is
that the basis-projection arithmetic is right.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tomcosmos.analysis.rotating_frame import (
    angular_position_relative_to,
    rotate_history_to_corotating,
)
from tomcosmos.io.history import StateHistory
from tomcosmos.state.scenario import Scenario


def _circular_two_body_history(
    n_samples: int = 600,
    period_s: float = 365.25 * 86400.0,
    R: float = 1.5e8,
    angle_off_l4: float = 0.0,
) -> StateHistory:
    """Three-body trajectory: stationary primary at origin, secondary
    on a circular orbit around it, test particle at L4 ± `angle_off_l4`
    radians (so 0 = exact L4, small positive = leading-side libration)."""
    t = np.linspace(0.0, period_s, n_samples)
    omega = 2.0 * np.pi / period_s
    theta = omega * t

    # Secondary's circular orbit in the xy-plane.
    sec_x = R * np.cos(theta)
    sec_y = R * np.sin(theta)
    sec_z = np.zeros_like(theta)

    # L4 leads the secondary by 60° in the orbit. Adding angle_off_l4
    # offsets the particle from L4 within the rotating frame.
    p_theta = theta + np.deg2rad(60.0) + angle_off_l4
    p_x = R * np.cos(p_theta)
    p_y = R * np.sin(p_theta)
    p_z = np.zeros_like(theta)

    rows = []
    for i, ti in enumerate(t):
        rows.append({"sample_idx": i, "t_tdb": ti, "body": "sun",
                     "x": 0.0, "y": 0.0, "z": 0.0,
                     "vx": 0.0, "vy": 0.0, "vz": 0.0,
                     "terminated": False, "energy_rel_err": 0.0})
        rows.append({"sample_idx": i, "t_tdb": ti, "body": "earth",
                     "x": sec_x[i], "y": sec_y[i], "z": sec_z[i],
                     "vx": -R * omega * np.sin(theta[i]),
                     "vy":  R * omega * np.cos(theta[i]),
                     "vz": 0.0,
                     "terminated": False, "energy_rel_err": 0.0})
        rows.append({"sample_idx": i, "t_tdb": ti, "body": "trojan",
                     "x": p_x[i], "y": p_y[i], "z": p_z[i],
                     "vx": 0.0, "vy": 0.0, "vz": 0.0,
                     "terminated": False, "energy_rel_err": 0.0})

    df = pd.DataFrame(rows)
    scenario = Scenario.model_validate({
        "schema_version": 1, "name": "rotating-frame-test",
        "epoch": "2026-01-01T00:00:00 TDB", "duration": "1 yr",
        "integrator": {"name": "ias15"},
        "output": {"format": "parquet", "cadence": "1 day"},
        "bodies": [
            {"name": "sun",   "spice_id": 10,  "mass_kg": 1.0e30,
             "radius_km": 1.0, "ic": {"source": "explicit",
                                      "r": [0, 0, 0], "v": [0, 0, 0]}},
            {"name": "earth", "spice_id": 399, "mass_kg": 1.0e24,
             "radius_km": 1.0, "ic": {"source": "explicit",
                                      "r": [R, 0, 0], "v": [0, 30.0, 0]}},
        ],
    })
    return StateHistory(df=df, scenario=scenario,
                        body_names=("sun", "earth", "trojan"))


def test_secondary_lands_at_positive_x_at_every_sample() -> None:
    """The defining property of the corotating frame: the secondary's
    rotated coordinates are (R, 0, 0) at every sample, regardless of
    its inertial-frame orbital phase."""
    history = _circular_two_body_history()
    rotated = rotate_history_to_corotating(history, primary="sun", secondary="earth")
    earth_rot = rotated["earth"]
    R = 1.5e8
    assert np.allclose(earth_rot[:, 0], R, atol=1e-3)
    assert np.allclose(earth_rot[:, 1], 0.0, atol=1e-3)
    assert np.allclose(earth_rot[:, 2], 0.0, atol=1e-3)


def test_l4_particle_lands_at_60_degrees() -> None:
    """A particle at exact L4 (60° ahead of the secondary in its orbit)
    should land at angular position +60° in the rotating frame xy-plane."""
    history = _circular_two_body_history(angle_off_l4=0.0)
    rotated = rotate_history_to_corotating(history, primary="sun", secondary="earth")
    delta = angular_position_relative_to(rotated, "trojan", reference_angle_deg=60.0)
    # All samples should be very close to 0° relative to L4.
    assert np.max(np.abs(delta)) < 1e-3, f"max libration {np.max(np.abs(delta))}°"


def test_libration_amplitude_recovered() -> None:
    """A particle offset from L4 by a known angle should produce that
    angle in the analysis output. Verifies the wrap-to-(-180, 180]
    arithmetic doesn't bite for sub-degree offsets."""
    history = _circular_two_body_history(angle_off_l4=np.deg2rad(2.5))
    rotated = rotate_history_to_corotating(history, primary="sun", secondary="earth")
    delta = angular_position_relative_to(rotated, "trojan", reference_angle_deg=60.0)
    assert abs(float(delta.mean()) - 2.5) < 1e-3
    assert abs(float(delta.std())) < 1e-3  # constant offset, no time variation


def test_unknown_body_rejected() -> None:
    history = _circular_two_body_history()
    with pytest.raises(KeyError):
        rotate_history_to_corotating(history, primary="venus", secondary="earth")


def test_primary_equals_secondary_rejected() -> None:
    history = _circular_two_body_history()
    with pytest.raises(ValueError, match="must differ"):
        rotate_history_to_corotating(history, primary="earth", secondary="earth")
