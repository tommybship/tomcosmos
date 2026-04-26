"""Pure two-body Keplerian mechanics — elements ↔ state vector.

Deliberately unit-agnostic: the caller chooses the consistent set
(`a` in km, `mu` in km³/s² → r in km, v in km/s; `a` in AU, `mu`
in AU³/yr² → r in AU, v in AU/yr). Angles in radians, time in the
matching unit.

Used by:
  - `state.ic._resolve_keplerian` — scenario-level Keplerian ICs
    around an explicit parent body.
  - `targeting.sbdb.state_at_epoch` — JPL SBDB elements →
    heliocentric (then barycentric) state vector for asteroid ICs.

Centralizing the math avoids two parallel Kepler implementations
drifting apart. The Newton solver and rotation matrix are
load-bearing for both code paths.
"""
from __future__ import annotations

import numpy as np


def solve_kepler_equation(M: float, e: float, tol: float = 1e-12) -> float:
    """Solve Kepler's equation E − e·sin E = M for the eccentric anomaly E.

    Uses Newton-Raphson with Danby's seed `E₀ = M + 0.85·e·sign(M)`,
    which converges quadratically for any eccentricity in [0, 1).
    `M` is wrapped to [-π, π] before iteration so the seed is always
    in a sensible neighborhood.
    """
    M = (M + np.pi) % (2.0 * np.pi) - np.pi
    E = M + 0.85 * e * (1.0 if M >= 0.0 else -1.0)
    for _ in range(50):
        f = E - e * np.sin(E) - M
        fp = 1.0 - e * np.cos(E)
        delta = f / fp
        E -= delta
        if abs(delta) < tol:
            return E
    raise RuntimeError(f"Kepler solver did not converge: M={M}, e={e}")


def perifocal_rotation_3_1_3(
    raan_rad: float, inc_rad: float, argp_rad: float,
) -> np.ndarray:
    """3-1-3 Euler rotation matrix R3(-Ω) · R1(-i) · R3(-ω).

    Standard astrodynamics transform that takes a vector from the
    perifocal frame (x along periapsis, z along orbital angular
    momentum) into the reference frame the orbital elements live in
    — typically ecliptic J2000 for SBDB elements, or ICRF if
    RAAN/inc/argp were measured against the equator.
    """
    cO, sO = np.cos(raan_rad), np.sin(raan_rad)
    ci, si = np.cos(inc_rad), np.sin(inc_rad)
    cw, sw = np.cos(argp_rad), np.sin(argp_rad)
    return np.array([
        [cO * cw - sO * sw * ci,  -cO * sw - sO * cw * ci,   sO * si],
        [sO * cw + cO * sw * ci,  -sO * sw + cO * cw * ci,  -cO * si],
        [sw * si,                  cw * si,                  ci    ],
    ])


def keplerian_to_state(
    *,
    a: float,
    e: float,
    inc_rad: float,
    raan_rad: float,
    argp_rad: float,
    mean_anom_rad: float,
    mu: float,
    dt: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Six classical elements + propagation interval → (r, v) in the
    elements' reference frame, centered on the parent body.

    Pipeline:
      1. Advance mean anomaly: `M(t) = M₀ + n·dt`, where n = sqrt(μ/a³).
      2. Solve Kepler's equation for the eccentric anomaly E.
      3. Build position+velocity in the perifocal (PQW) frame from E.
      4. Rotate by 3-1-3 Euler (Ω, i, ω) into the reference frame.

    Caller owns the lift from "centered on parent body" to whatever
    inertial origin the integration runs in — typically `r_parent +
    R · r_pqw`, with the parent's state provided by an ephemeris
    source.

    Units: `a` and `mu` must form a consistent set (e.g. km + km³/s²,
    or AU + AU³/yr²); `dt` matches `mu`'s time unit. Angles in radians.
    """
    n = float(np.sqrt(mu / a ** 3))
    M = mean_anom_rad + n * dt
    E = solve_kepler_equation(M, e)

    cos_E, sin_E = np.cos(E), np.sin(E)
    sqrt_one_minus_e2 = np.sqrt(1.0 - e ** 2)

    # Perifocal frame: x toward periapsis, y 90° ahead in orbit, z along
    # orbital angular momentum.
    r_pqw = np.array([
        a * (cos_E - e),
        a * sqrt_one_minus_e2 * sin_E,
        0.0,
    ])
    rdot_factor = float(np.sqrt(mu * a)) / (a * (1.0 - e * cos_E))
    v_pqw = rdot_factor * np.array([
        -sin_E,
        sqrt_one_minus_e2 * cos_E,
        0.0,
    ])

    R = perifocal_rotation_3_1_3(raan_rad, inc_rad, argp_rad)
    return R @ r_pqw, R @ v_pqw
