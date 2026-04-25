"""Lambert solver tests.

Two verification angles:

1. Closed-form Hohmann transfer (coplanar circular orbits, 180° transfer
   angle). Δv at periapsis and apoapsis match the textbook
   `vc·(√(2r₂/(r₁+r₂)) − 1)` expressions.
2. Round-trip propagation: solving Lambert(r₁, r₂, tof) and then
   integrating from (r₁, v₁) for `tof` should land at r₂ to numerical
   precision. Independent of any specific closed form — just verifies
   that the returned velocities are dynamically self-consistent.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tomcosmos.targeting import lambert

# Sun's gravitational parameter in km³/s².
MU_SUN = 1.32712440018e11
AU_KM = 1.495978707e8


def test_near_hohmann_earth_to_mars_matches_closed_form() -> None:
    """Near-Hohmann Earth → Mars transfer (Δν = 180° − 0.5°).

    r₁ = 1 AU, r₂ = 1.524 AU at slight tilt off the antiparallel line.
    Δν = 180° exactly is the canonical Lambert singularity (transfer
    plane is undefined when r₁ and r₂ are colinear with the focus), so
    we tilt by 0.5° to recover a well-defined plane. The closed-form
    Hohmann Δv values differ from the tilted case by O(ε), well below
    the 0.5% test tolerance.

    Closed form (true Δν = 180°):
      v_periapsis = √(μ/r₁) · √(2r₂/(r₁+r₂))
      v_apoapsis  = √(μ/r₂) · √(2r₁/(r₁+r₂))
      tof         = π · √(((r₁+r₂)/2)³ / μ)
    """
    r1_km = 1.0 * AU_KM
    r2_km = 1.524 * AU_KM
    a_transfer = 0.5 * (r1_km + r2_km)
    tof_s = math.pi * math.sqrt(a_transfer ** 3 / MU_SUN)

    # r₁ at +x; r₂ at angle 180° − 0.5° from r₁ (tilted slightly off
    # the antiparallel singularity).
    angle = math.pi - math.radians(0.5)
    r1 = np.array([r1_km, 0.0, 0.0])
    r2 = np.array([r2_km * math.cos(angle), r2_km * math.sin(angle), 0.0])

    v1, v2 = lambert(r1, r2, tof_s, MU_SUN, prograde=True)

    vc1 = math.sqrt(MU_SUN / r1_km)
    vc2 = math.sqrt(MU_SUN / r2_km)
    v1_expected = vc1 * math.sqrt(2.0 * r2_km / (r1_km + r2_km))
    v2_expected = vc2 * math.sqrt(2.0 * r1_km / (r1_km + r2_km))

    # 0.5° tilt → ~5e-3 relative deviation from the perfect Hohmann.
    assert abs(np.linalg.norm(v1) - v1_expected) / v1_expected < 5e-3
    assert abs(np.linalg.norm(v2) - v2_expected) / v2_expected < 5e-3

    # Direction: prograde transfer with r₁ at +x sends v₁ mostly in +y.
    assert v1[1] > 0
    assert abs(v1[2]) < 1e-9 and abs(v2[2]) < 1e-9  # planar transfer


def test_hohmann_delta_v_matches_textbook() -> None:
    """The Δv Lambert hands back, when subtracted from a pre-burn
    circular velocity, matches the textbook Hohmann departure-burn
    formula `vc·(√(2r₂/(r₁+r₂)) − 1)` within ~1% (the slight tilt
    away from the Δν=180° singularity is the residual)."""
    r1_km = 1.0 * AU_KM
    r2_km = 1.524 * AU_KM
    a_transfer = 0.5 * (r1_km + r2_km)
    tof_s = math.pi * math.sqrt(a_transfer ** 3 / MU_SUN)

    angle = math.pi - math.radians(0.5)
    r1 = np.array([r1_km, 0.0, 0.0])
    r2 = np.array([r2_km * math.cos(angle), r2_km * math.sin(angle), 0.0])
    v1, _ = lambert(r1, r2, tof_s, MU_SUN, prograde=True)

    v_circ1 = np.array([0.0, math.sqrt(MU_SUN / r1_km), 0.0])
    delta_v_dep = v1 - v_circ1
    expected_dep_kms = math.sqrt(MU_SUN / r1_km) * (
        math.sqrt(2.0 * r2_km / (r1_km + r2_km)) - 1.0
    )
    rel_err = abs(np.linalg.norm(delta_v_dep) - expected_dep_kms) / expected_dep_kms
    assert rel_err < 1e-2  # ~0.5° tilt residual


def test_lambert_round_trip_propagation() -> None:
    """Propagate (r₁, v₁) for tof under Sun gravity (REBOUND two-body)
    and verify we land at r₂. Independent confirmation that Lambert's
    output is dynamically consistent."""
    import rebound
    # 90° transfer angle, 1 AU starting radius, 1 AU ending radius.
    # Tof ≈ 1/4 of the circular period (≈ 91.3 days for r=1 AU).
    r1 = np.array([1.0 * AU_KM, 0.0, 0.0])
    r2 = np.array([0.0, 1.0 * AU_KM, 0.0])
    tof_s = 0.5 * 2.0 * math.pi * math.sqrt((1.0 * AU_KM) ** 3 / MU_SUN) / 2

    v1, _v2 = lambert(r1, r2, tof_s, MU_SUN, prograde=True)

    # Set up REBOUND in a SI-derived unit system (km, s, kg) for direct
    # comparison. Use a Sun particle and a probe; integrate the probe.
    sim = rebound.Simulation()
    sim.units = ('m', 's', 'kg')
    sim.G = 6.6743e-11
    M_sun_kg = MU_SUN * 1e9 / sim.G  # μ km³/s² → m³/s² → mass
    sim.add(m=M_sun_kg, x=0, y=0, z=0, vx=0, vy=0, vz=0)
    sim.add(
        m=0.0,
        x=r1[0] * 1e3, y=r1[1] * 1e3, z=r1[2] * 1e3,
        vx=v1[0] * 1e3, vy=v1[1] * 1e3, vz=v1[2] * 1e3,
    )
    sim.move_to_com()
    sim.integrator = "ias15"
    sim.integrate(tof_s)

    p = sim.particles[1]
    arrived_km = np.array([p.x, p.y, p.z]) / 1e3
    # IAS15 over a quarter-orbit at AU scale → sub-km accuracy.
    err_km = float(np.linalg.norm(arrived_km - r2))
    assert err_km < 1.0, f"propagation off-target by {err_km:.3e} km"


def test_lambert_long_way_flips_velocity_direction() -> None:
    """Same r₁, r₂, tof but with `prograde=False` — the long-way
    transfer crosses through the side of the focus opposite the
    short-way arc, so v₁ should point in the opposite tangential
    direction."""
    r1_km = 1.0 * AU_KM
    r2_km = 1.524 * AU_KM
    # 90° transfer angle, well-defined plane regardless of direction.
    tof_s = 0.5 * 365.25 * 86400.0  # half a year — single-revolution
    r1 = np.array([r1_km, 0.0, 0.0])
    r2 = np.array([0.0, r2_km, 0.0])

    v1_pro, _ = lambert(r1, r2, tof_s, MU_SUN, prograde=True)
    v1_ret, _ = lambert(r1, r2, tof_s, MU_SUN, prograde=False)
    # Prograde and retrograde transfers connect the same endpoints with
    # opposite-signed tangential velocities. Specifically, the projection
    # of v₁ onto the tangent direction at r₁ flips sign.
    tangent = np.array([0.0, 1.0, 0.0])  # +y is the prograde tangent at r₁
    assert float(np.dot(v1_pro, tangent)) > 0
    assert float(np.dot(v1_ret, tangent)) < 0


def test_lambert_rejects_nonpositive_tof() -> None:
    r1 = np.array([AU_KM, 0.0, 0.0])
    r2 = np.array([0.0, AU_KM, 0.0])
    with pytest.raises(ValueError, match="tof_s must be positive"):
        lambert(r1, r2, 0.0, MU_SUN)
    with pytest.raises(ValueError, match="tof_s must be positive"):
        lambert(r1, r2, -1.0, MU_SUN)


def test_lambert_rejects_zero_position() -> None:
    r1 = np.array([0.0, 0.0, 0.0])
    r2 = np.array([0.0, AU_KM, 0.0])
    with pytest.raises(ValueError, match="must be non-zero"):
        lambert(r1, r2, 1e6, MU_SUN)


def test_lambert_rejects_colinear_geometry() -> None:
    """r₁ and r₂ on the same ray from the focus → Δν=0; Lambert is
    degenerate. Same goes for Δν=π exactly (vector along opposite ray);
    here the transfer plane is undefined but our test angle 180° still
    has a definite plane (z=0), so we just check the Δν=0 case."""
    r1 = np.array([AU_KM, 0.0, 0.0])
    r2 = np.array([2.0 * AU_KM, 0.0, 0.0])
    with pytest.raises(ValueError, match="colinear"):
        lambert(r1, r2, 1e7, MU_SUN)
