"""Lambert's problem — find the orbit connecting two position vectors
in a specified time of flight.

Given r₁ (start position), r₂ (end position), and tof (time of flight),
plus the gravitational parameter μ of the central body, return the
velocity vectors v₁ at start and v₂ at end. Uniquely determines the
two-body Keplerian arc that takes you from (r₁, t₁) to (r₂, t₁ + tof)
under the central body's gravity alone.

This implementation uses the universal-variable formulation from
Bate-Mueller-White (1971), as presented in Vallado §7.6. It handles
single-revolution elliptic, parabolic, and hyperbolic transfers in one
unified Newton iteration on the universal anomaly variable z.
Multi-revolution Lambert (multiple orbits between r₁ and r₂) is not
supported here — for that, switch to Izzo's algorithm. Single-rev
covers Hohmann-class transfers, slingshot setup arcs, and most
mission-design educational use cases.

References:
  Bate, Mueller, White, "Fundamentals of Astrodynamics" (1971), §5.3.
  Vallado, "Fundamentals of Astrodynamics and Applications", 4th ed., §7.6.
"""
from __future__ import annotations

import math

import numpy as np

# Newton iteration limits. The bracketing logic in `lambert()` keeps z in
# a regime where C(z) > 0 and y(z) > 0; once that's satisfied the
# Newton update converges quadratically and these caps almost always
# bottom out on the absolute-tolerance test below.
_MAX_NEWTON_ITERS = 200
_TOF_REL_TOL = 1e-10
_Z_ABS_FLOOR = 1e-12  # below this we use the z=0 limit forms


def lambert(
    r1: np.ndarray,
    r2: np.ndarray,
    tof_s: float,
    mu: float,
    *,
    prograde: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve Lambert's problem for a single-revolution transfer.

    Parameters
    ----------
    r1, r2 : (3,) arrays
        Position vectors at start and end, in any consistent length units
        (km is the tomcosmos convention). Reference frame must match `mu`.
    tof_s : float
        Time of flight from r₁ to r₂ in seconds.
    mu : float
        Standard gravitational parameter of the central body, in
        (length-unit)³ / s² (i.e. km³/s² when r₁, r₂ are in km).
    prograde : bool, default True
        Orbit direction. Prograde = z-component of (r₁ × r₂) positive,
        i.e. the transfer winds counterclockwise as viewed from +z. For
        Earth → Mars in the ecliptic frame, prograde is the natural
        choice (planets orbit prograde).

    Returns
    -------
    (v1, v2) : tuple of (3,) arrays
        Velocity vectors at r₁ and r₂ in (length-unit)/s. Add v1 to the
        spacecraft's pre-burn velocity to get the departure Δv; subtract
        v2 from the target's velocity at arrival to get the arrival Δv.
    """
    r1 = np.asarray(r1, dtype=np.float64)
    r2 = np.asarray(r2, dtype=np.float64)
    if r1.shape != (3,) or r2.shape != (3,):
        raise ValueError(f"r1 and r2 must have shape (3,); got {r1.shape}, {r2.shape}")
    if tof_s <= 0:
        raise ValueError(f"tof_s must be positive; got {tof_s}")
    if mu <= 0:
        raise ValueError(f"mu must be positive; got {mu}")

    R1 = float(np.linalg.norm(r1))
    R2 = float(np.linalg.norm(r2))
    if R1 == 0.0 or R2 == 0.0:
        raise ValueError("r1 and r2 must be non-zero")

    # Transfer angle Δν (true-anomaly change) and direction sign.
    cos_dnu = float(np.dot(r1, r2)) / (R1 * R2)
    cos_dnu = max(-1.0, min(1.0, cos_dnu))  # clip against fp drift
    cross_z = float(r1[0] * r2[1] - r1[1] * r2[0])

    # `prograde` is interpreted in the +z-up sense. If the natural cross
    # product disagrees with the requested direction, take Δν as the
    # reflex (long-way) angle so sin(Δν) gets the right sign.
    if prograde:
        sin_dnu = math.sqrt(1.0 - cos_dnu * cos_dnu)
        if cross_z < 0.0:
            sin_dnu = -sin_dnu
    else:
        sin_dnu = -math.sqrt(1.0 - cos_dnu * cos_dnu)
        if cross_z < 0.0:
            sin_dnu = -sin_dnu

    if abs(sin_dnu) < 1e-15:
        raise ValueError(
            "r1 and r2 are colinear with the focus (Δν ≈ 0 or π); "
            "Lambert is degenerate and the transfer plane is undefined"
        )

    # A is the geometry constant from Vallado eq. 7-71. It captures the
    # chord length and direction in a form that simplifies the universal-
    # variable iteration below.
    A = sin_dnu * math.sqrt(R1 * R2 / (1.0 - cos_dnu))

    z, y = _solve_for_z(R1, R2, A, tof_s, mu)

    # Lagrange coefficients (Vallado eq. 7-74). f and g express r₂ as a
    # linear combination of r₁ and v₁; g_dot does the same for v₂.
    f = 1.0 - y / R1
    g = A * math.sqrt(y / mu)
    g_dot = 1.0 - y / R2

    v1 = (r2 - f * r1) / g
    v2 = (g_dot * r2 - r1) / g
    return v1, v2


# --- Internals -------------------------------------------------------------


def _solve_for_z(R1: float, R2: float, A: float, tof_s: float, mu: float) -> tuple[float, float]:
    """Iterate on the universal anomaly variable z to match `tof_s`.

    Uses a secant method on the residual T(z) − tof_s. T(z) is
    monotonically increasing in z over the feasible domain (where
    y(z) > 0), so the secant root-finder converges robustly without
    needing the closed-form Stumpff derivatives.

    Returns the converged (z, y) pair. y = R₁ + R₂ + A·(z·S(z) − 1)/√C(z)
    is needed by the velocity reconstruction in `lambert()`, so we hand
    it back rather than recomputing.
    """
    # Bracket the feasible region first: pick two z values where y(z) > 0.
    # Below z_lo the transfer is too fast (hyperbolic, y < 0 territory);
    # above z_hi we're deep in the slow-elliptic regime. The bracket
    # below covers Hohmann-like transfers (z near 0) through several
    # orbital periods of slow-elliptic.
    z_lo, z_hi = -16.0, 4.0 * math.pi ** 2
    # Walk z_lo upward until y becomes positive.
    while _y_of_z(R1, R2, A, z_lo) <= 0:
        z_lo += 0.5
        if z_lo >= z_hi:  # pragma: no cover — pathological geometry
            raise RuntimeError(
                "Lambert: could not find a feasible z bracket; geometry "
                "may not admit a single-revolution transfer at this tof"
            )

    def _residual(z: float) -> float:
        y = _y_of_z(R1, R2, A, z)
        if y < 0:
            return float("-inf")  # below feasible region
        x = math.sqrt(y / _C(z))
        return (x ** 3 * _S(z) + A * math.sqrt(y)) / math.sqrt(mu) - tof_s

    # Secant iteration with safeguarding: clamp into the feasible bracket
    # if a step would take us out. Two prior points seed the secant.
    z_prev, z_curr = z_lo, z_lo + 0.5
    f_prev = _residual(z_prev)
    f_curr = _residual(z_curr)
    for _ in range(_MAX_NEWTON_ITERS):
        if abs(f_curr) < _TOF_REL_TOL * tof_s:
            y = _y_of_z(R1, R2, A, z_curr)
            return z_curr, y

        if f_curr == f_prev:  # pragma: no cover — secant denominator
            raise RuntimeError("Lambert secant: zero denominator")
        z_next = z_curr - f_curr * (z_curr - z_prev) / (f_curr - f_prev)
        # Safeguard: stay above the feasibility floor.
        if _y_of_z(R1, R2, A, z_next) <= 0:
            z_next = 0.5 * (z_curr + max(z_lo, z_next))

        z_prev, z_curr = z_curr, z_next
        f_prev, f_curr = f_curr, _residual(z_curr)

    raise RuntimeError(
        f"Lambert secant iteration did not converge after {_MAX_NEWTON_ITERS} steps "
        f"(last z={z_curr}, last residual={f_curr})"
    )


def _y_of_z(R1: float, R2: float, A: float, z: float) -> float:
    """y(z) per Vallado eq. 7-72: orbit-radius parameter at universal
    anomaly z. Negative y means the geometry is infeasible at this z."""
    return R1 + R2 + A * (z * _S(z) - 1.0) / math.sqrt(_C(z))


def _C(z: float) -> float:
    """Stumpff C(z): (1 − cos√z)/z for z>0, (cosh√−z − 1)/(−z) for z<0,
    1/2 at z=0. Continuous and analytic across the parabolic boundary."""
    if z > _Z_ABS_FLOOR:
        sq = math.sqrt(z)
        return (1.0 - math.cos(sq)) / z
    if z < -_Z_ABS_FLOOR:
        sq = math.sqrt(-z)
        return (math.cosh(sq) - 1.0) / (-z)
    # Taylor expansion around 0: C(z) ≈ 1/2 − z/24 + z²/720 − ...
    return 0.5 - z / 24.0 + z * z / 720.0


def _S(z: float) -> float:
    """Stumpff S(z): (√z − sin√z)/z^(3/2) for z>0, analogous for z<0,
    1/6 at z=0."""
    if z > _Z_ABS_FLOOR:
        sq = math.sqrt(z)
        return (sq - math.sin(sq)) / (z ** 1.5)
    if z < -_Z_ABS_FLOOR:
        sq = math.sqrt(-z)
        return (math.sinh(sq) - sq) / ((-z) ** 1.5)
    # Taylor expansion around 0: S(z) ≈ 1/6 − z/120 + z²/5040 − ...
    return 1.0 / 6.0 - z / 120.0 + z * z / 5040.0


