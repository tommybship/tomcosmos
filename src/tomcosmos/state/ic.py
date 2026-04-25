"""Initial-condition resolution.

Turns declarative `Body` / `TestParticle` scenario entries into concrete
physics inputs: (name, mass_kg, radius_km, r_km, v_kms) all expressed
in ICRF barycentric. The integrator (state/integrator.py) consumes these
directly — it never looks at the YAML-level types.

Test-particle IC types:
    TestParticleExplicitIc  — r, v in any supported frame
    TestParticleLagrangeIc  — L1-L5 of a primary/secondary pair
    TestParticleKeplerianIc — six-element osculating orbit around a parent
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from astropy import units as u
from astropy.time import Time

from tomcosmos.constants import BodyConstant, resolve_body_constant
from tomcosmos.exceptions import ScenarioValidationError, UnknownBodyError
from tomcosmos.state.ephemeris import EphemerisSource
from tomcosmos.state.frames import ecliptic_to_icrf
from tomcosmos.state.scenario import (
    Body,
    EphemerisIc,
    ExplicitIc,
    Scenario,
    TestParticle,
    TestParticleExplicitIc,
    TestParticleKeplerianIc,
    TestParticleLagrangeIc,
)

# Gravitational constant in km³/(kg·s²). Same convention as the rest of
# tomcosmos's I/O boundary (km, km/s).
_G_KM3_PER_KG_S2 = 6.6743e-20


@dataclass(frozen=True)
class ResolvedBody:
    """Everything the integrator needs to add one massive body to REBOUND."""

    name: str
    mass_kg: float
    radius_km: float
    r_km: np.ndarray          # shape (3,), ICRF barycentric
    v_kms: np.ndarray         # shape (3,), ICRF barycentric
    spice_id: int | None = None
    color_hex: str | None = None


@dataclass(frozen=True)
class ResolvedTestParticle:
    """Massless particle — position + velocity, name for hash identity."""

    name: str
    r_km: np.ndarray
    v_kms: np.ndarray


def resolve_scenario(
    scenario: Scenario, source: EphemerisSource
) -> tuple[list[ResolvedBody], list[ResolvedTestParticle]]:
    """Resolve all bodies and test particles at the scenario epoch.

    Requires the ephemeris source to cover `scenario.epoch + scenario.duration`;
    the caller should have already validated that via `source.require_covers()`.
    """
    bodies = [resolve_body(b, scenario.epoch, source) for b in scenario.bodies]
    particles = [
        resolve_test_particle(p, scenario.epoch, source, scenario)
        for p in scenario.test_particles
    ]
    return bodies, particles


def resolve_body(
    body: Body, epoch: Time, source: EphemerisSource
) -> ResolvedBody:
    constant = _lookup_constant(body)

    mass_kg = body.mass_kg if body.mass_kg is not None else _require_field(
        constant, body.name, "mass_kg"
    )
    radius_km = body.radius_km if body.radius_km is not None else _require_field(
        constant, body.name, "radius_km"
    )

    if isinstance(body.ic, EphemerisIc):
        key: str | int = body.spice_id if body.spice_id is not None else body.name
        r, v = source.query(key, epoch)
    elif isinstance(body.ic, ExplicitIc):
        r, v = _to_icrf_barycentric(
            body.ic.r, body.ic.v, body.ic.frame, epoch, source
        )
    else:  # pragma: no cover — exhaustive over the discriminated union
        raise ScenarioValidationError(
            f"body {body.name!r}: unsupported ic type {type(body.ic).__name__}"
        )

    return ResolvedBody(
        name=body.name,
        mass_kg=mass_kg,
        radius_km=radius_km,
        r_km=np.asarray(r, dtype=np.float64),
        v_kms=np.asarray(v, dtype=np.float64),
        spice_id=(body.spice_id if body.spice_id is not None
                  else (constant.spice_id if constant is not None else None)),
        color_hex=constant.color_hex if constant is not None else None,
    )


def resolve_test_particle(
    particle: TestParticle,
    epoch: Time,
    source: EphemerisSource,
    scenario: Scenario | None = None,
) -> ResolvedTestParticle:
    """Resolve a test particle's IC to ICRF barycentric (r_km, v_kms).

    `scenario` is required for Lagrange and Keplerian IC types because
    they reference other bodies by name. Explicit ICs ignore it.
    """
    if isinstance(particle.ic, TestParticleExplicitIc):
        r, v = _to_icrf_barycentric(
            particle.ic.r, particle.ic.v, particle.ic.frame, epoch, source
        )
    elif isinstance(particle.ic, TestParticleLagrangeIc):
        if scenario is None:
            raise ScenarioValidationError(
                f"test particle {particle.name!r}: lagrange IC requires "
                "a Scenario context to resolve primary/secondary"
            )
        r, v = _resolve_lagrange(particle.ic, scenario, epoch, source)
    elif isinstance(particle.ic, TestParticleKeplerianIc):
        if scenario is None:
            raise ScenarioValidationError(
                f"test particle {particle.name!r}: keplerian IC requires "
                "a Scenario context to resolve parent"
            )
        r, v = _resolve_keplerian(particle.ic, scenario, epoch, source)
    else:  # pragma: no cover — exhaustive over the discriminated union
        raise ScenarioValidationError(
            f"test particle {particle.name!r}: "
            f"unsupported ic type {type(particle.ic).__name__}"
        )
    return ResolvedTestParticle(
        name=particle.name,
        r_km=np.asarray(r, dtype=np.float64),
        v_kms=np.asarray(v, dtype=np.float64),
    )


def _lookup_constant(body: Body) -> BodyConstant | None:
    """Try SPICE ID first, then canonical name. Missing lookup is not fatal
    here — it becomes fatal only if mass or radius are also missing."""
    if body.spice_id is not None:
        try:
            return resolve_body_constant(body.spice_id)
        except UnknownBodyError:
            pass
    try:
        return resolve_body_constant(body.name)
    except UnknownBodyError:
        return None


def _require_field(
    constant: BodyConstant | None, body_name: str, field_name: str
) -> float:
    if constant is None:
        raise ScenarioValidationError(
            f"body {body_name!r}: {field_name} not provided and name/spice_id "
            f"is not in constants.BODY_CONSTANTS"
        )
    return float(getattr(constant, field_name))


def _to_icrf_barycentric(
    r: tuple[float, float, float] | np.ndarray,
    v: tuple[float, float, float] | np.ndarray,
    frame: str,
    epoch: Time,
    source: EphemerisSource,
) -> tuple[np.ndarray, np.ndarray]:
    r_arr = np.asarray(r, dtype=np.float64)
    v_arr = np.asarray(v, dtype=np.float64)

    if frame == "icrf_barycentric":
        return r_arr, v_arr
    if frame == "ecliptic_j2000_barycentric":
        return ecliptic_to_icrf(r_arr), ecliptic_to_icrf(v_arr)
    if frame == "icrf_heliocentric":
        r_sun, v_sun = source.query("sun", epoch)
        return r_arr + r_sun, v_arr + v_sun
    if frame == "ecliptic_j2000_heliocentric":
        r_icrf = ecliptic_to_icrf(r_arr)
        v_icrf = ecliptic_to_icrf(v_arr)
        r_sun, v_sun = source.query("sun", epoch)
        return r_icrf + r_sun, v_icrf + v_sun

    # Scenario schema already rejects unknown frames at validation time, but
    # if a caller builds a Scenario in-memory with a bogus string this is the
    # safety net.
    raise ScenarioValidationError(f"unsupported frame: {frame!r}")


# ---------------------------------------------------------------------------
# Lagrange-point IC
# ---------------------------------------------------------------------------


def _resolve_lagrange(
    ic: TestParticleLagrangeIc,
    scenario: Scenario,
    epoch: Time,
    source: EphemerisSource,
) -> tuple[np.ndarray, np.ndarray]:
    """Place a particle at L1-L5 of (primary, secondary).

    Math:
      Mass ratio μ = m_2 / (m_1 + m_2). The line through primary and
      secondary defines the rotating frame's x-axis; angular velocity
      ω points along the orbital angular momentum z. L4/L5 sit at the
      apex of an equilateral triangle with the two masses (±60°
      ahead/behind the secondary in its orbit). L1/L2/L3 are roots of
      a quintic in normalized units; we solve numerically.

    Then we transform back to ICRF barycentric:
      r_icrf = r_primary + R · r_rotating
      v_icrf = v_primary + ω × (R · r_rotating) + R · 0
    where R rotates the rotating-frame x-axis onto the primary→secondary
    line. The "+ R · 0" term is the rotating-frame velocity, which is
    zero at any Lagrange point by definition (they're equilibria).
    """
    primary_body = _scenario_body_by_name(scenario, ic.primary, "primary")
    secondary_body = _scenario_body_by_name(scenario, ic.secondary, "secondary")

    r1, v1 = _state_for(primary_body, epoch, source)
    r2, v2 = _state_for(secondary_body, epoch, source)
    m1 = _mass_kg_for(primary_body)
    m2 = _mass_kg_for(secondary_body)

    sep = r2 - r1
    R_sep = float(np.linalg.norm(sep))
    if R_sep <= 0.0:
        raise ScenarioValidationError(
            f"lagrange IC: primary {ic.primary!r} and secondary "
            f"{ic.secondary!r} share a position at epoch"
        )
    e_x = sep / R_sep

    # Angular momentum direction = orbital plane normal.
    rel_v = v2 - v1
    L = np.cross(sep, rel_v)
    L_norm = float(np.linalg.norm(L))
    if L_norm <= 0.0:
        raise ScenarioValidationError(
            f"lagrange IC: relative velocity of {ic.secondary!r} is "
            f"zero or parallel to separation; orbital plane undefined"
        )
    e_z = L / L_norm
    e_y = np.cross(e_z, e_x)

    mu = m2 / (m1 + m2)
    # Primary-centered rotating-frame basis (e_x, e_y, e_z): primary at
    # origin, secondary at (R_sep, 0, 0). r_rot is the Lagrange-point
    # vector relative to the primary.
    if ic.point in ("L4", "L5"):
        # Equilateral apex, ±60° from the secondary's position vector.
        # In primary-centered coords: x = R/2, |y| = (√3/2)·R.
        # L4 leads the secondary (positive y by orbital-angular-momentum
        # convention); L5 trails.
        x_rot = 0.5 * R_sep
        y_rot = (np.sqrt(3.0) / 2.0) * R_sep * (1.0 if ic.point == "L4" else -1.0)
        r_rot = x_rot * e_x + y_rot * e_y
    else:
        # Collinear points. `_lagrange_collinear_distance` returns the
        # dimensionless distance γ in normalized units of R from the
        # *adjacent* mass:
        #   L1: γ from secondary toward primary  → r = (1-γ)·R from primary
        #   L2: γ from secondary outward          → r = (1+γ)·R from primary
        #   L3: γ from primary on opposite side   → r = -γ·R from primary
        gamma = _lagrange_collinear_distance(ic.point, mu)
        if ic.point == "L1":
            x_rot = (1.0 - gamma) * R_sep
        elif ic.point == "L2":
            x_rot = (1.0 + gamma) * R_sep
        else:  # L3
            x_rot = -gamma * R_sep
        r_rot = x_rot * e_x

    r_icrf = r1 + r_rot

    # Velocity at a Lagrange point in the rotating frame is zero. In the
    # inertial frame, the particle co-rotates with the secondary:
    #   v_inertial = v_primary + ω × (r - r_primary)
    # ω = (rel_v × sep) / |sep|² · 0  ... no, ω is along e_z with
    # magnitude n = |L|/|sep|² (specific angular momentum / r²).
    omega_mag = L_norm / (R_sep * R_sep)
    omega = omega_mag * e_z
    v_icrf = v1 + np.cross(omega, r_icrf - r1)

    return r_icrf, v_icrf


def _lagrange_collinear_distance(point: str, mu: float) -> float:
    """Dimensionless distance |γ| from the relevant primary, in units of R.

    L1: between m1 and m2, distance γ from m2 (toward m1)
    L2: beyond m2 on the far side, distance γ from m2 (away from m1)
    L3: beyond m1 on the opposite side, distance γ from m1

    Quintics from Murray & Dermott §3.4. Solved with scipy.optimize
    isn't available here, so we use Newton-Raphson with a known good
    seed (the order-1 expansion in μ^(1/3)).
    """
    cube_root_third_mu = (mu / 3.0) ** (1.0 / 3.0)
    if point == "L1":
        # f(γ) = γ⁵ - (3-μ)γ⁴ + (3-2μ)γ³ - μγ² + 2μγ - μ = 0
        coeffs = (1.0, -(3.0 - mu), 3.0 - 2.0 * mu, -mu, 2.0 * mu, -mu)
        seed = cube_root_third_mu
    elif point == "L2":
        # f(γ) = γ⁵ + (3-μ)γ⁴ + (3-2μ)γ³ - μγ² - 2μγ - μ = 0
        coeffs = (1.0, 3.0 - mu, 3.0 - 2.0 * mu, -mu, -2.0 * mu, -mu)
        seed = cube_root_third_mu
    elif point == "L3":
        # f(γ) = γ⁵ + (2+μ)γ⁴ + (1+2μ)γ³ - (1-μ)γ² - 2(1-μ)γ - (1-μ) = 0
        coeffs = (
            1.0, 2.0 + mu, 1.0 + 2.0 * mu, -(1.0 - mu),
            -2.0 * (1.0 - mu), -(1.0 - mu),
        )
        seed = 1.0 - 7.0 * mu / 12.0  # M&D eq 3.83
    else:
        raise ValueError(f"_lagrange_collinear_distance: bad point {point!r}")
    return _newton_polynomial(coeffs, seed)


def _newton_polynomial(coeffs: tuple[float, ...], x0: float) -> float:
    """Solve poly(x)=0 from `x0` via Newton-Raphson. coeffs high-degree first.

    Used only for the L1/L2/L3 quintics, which have well-isolated
    real roots near the seed. Converges in ~5 iters; 50 is generous.
    """
    p = np.poly1d(coeffs)
    dp = p.deriv()
    x = x0
    for _ in range(50):
        f = float(p(x))
        if abs(f) < 1e-14:
            return x
        x -= f / float(dp(x))
    raise RuntimeError(f"Newton did not converge for coeffs={coeffs}")


# ---------------------------------------------------------------------------
# Keplerian-elements IC
# ---------------------------------------------------------------------------


def _resolve_keplerian(
    ic: TestParticleKeplerianIc,
    scenario: Scenario,
    epoch: Time,
    source: EphemerisSource,
) -> tuple[np.ndarray, np.ndarray]:
    """Six classical elements → ICRF barycentric (r_km, v_kms).

    Pipeline:
      1. Solve Kepler's equation E − e·sin E = M for eccentric anomaly E.
      2. Build position+velocity in the perifocal (PQW) frame.
      3. Rotate by R3(-Ω)·R1(-i)·R3(-ω) into the parent body's
         orbital frame (ecliptic or ICRF, per `ic.frame`).
      4. Add the parent body's ICRF state to lift to barycentric.
    """
    parent_body = _scenario_body_by_name(scenario, ic.parent, "parent")
    r_parent, v_parent = _state_for(parent_body, epoch, source)
    mu_parent = _G_KM3_PER_KG_S2 * _mass_kg_for(parent_body)

    # Kepler-propagate the mean anomaly if elements were measured at a
    # different epoch (M5 reuse — for now `epoch_offset_s == 0` is the
    # common path).
    n = float(np.sqrt(mu_parent / ic.a_km ** 3))  # rad/s
    M = np.deg2rad(ic.mean_anom_deg) + n * ic.epoch_offset_s
    E = _solve_kepler(M, ic.e)

    cos_E, sin_E = np.cos(E), np.sin(E)
    one_minus_e_cos_E = 1.0 - ic.e * cos_E

    # Perifocal coordinates: x along periapsis, y 90° ahead in the orbit.
    r_pqw = np.array([
        ic.a_km * (cos_E - ic.e),
        ic.a_km * np.sqrt(1.0 - ic.e ** 2) * sin_E,
        0.0,
    ])
    rdot_factor = float(np.sqrt(mu_parent * ic.a_km)) / (ic.a_km * one_minus_e_cos_E)
    v_pqw = rdot_factor * np.array([
        -sin_E,
        np.sqrt(1.0 - ic.e ** 2) * cos_E,
        0.0,
    ])

    # Perifocal → orbital plane (ecliptic or ICRF, per ic.frame).
    R = _euler_3_1_3(
        np.deg2rad(ic.raan_deg),
        np.deg2rad(ic.inc_deg),
        np.deg2rad(ic.argp_deg),
    )
    r_in_frame = R @ r_pqw
    v_in_frame = R @ v_pqw

    # Lift to ICRF barycentric (frame and parent both contribute).
    if ic.frame == "ecliptic_j2000_barycentric":
        r_in_frame = ecliptic_to_icrf(r_in_frame)
        v_in_frame = ecliptic_to_icrf(v_in_frame)
    return r_parent + r_in_frame, v_parent + v_in_frame


def _solve_kepler(M: float, e: float, tol: float = 1e-12) -> float:
    """Newton-Raphson on E − e·sin E = M, with M wrapped to [-π, π]."""
    M = (M + np.pi) % (2.0 * np.pi) - np.pi
    # Danby's seed: works well for any eccentricity in [0, 1).
    E = M + 0.85 * e * (1.0 if M >= 0.0 else -1.0)
    for _ in range(50):
        f = E - e * np.sin(E) - M
        fp = 1.0 - e * np.cos(E)
        delta = f / fp
        E -= delta
        if abs(delta) < tol:
            return E
    raise RuntimeError(f"Kepler solver did not converge: M={M}, e={e}")


def _euler_3_1_3(raan: float, inc: float, argp: float) -> np.ndarray:
    """3-1-3 Euler rotation R3(-Ω) · R1(-i) · R3(-ω). Standard astrodynamics
    perifocal → reference-frame transform."""
    cO, sO = np.cos(raan), np.sin(raan)
    ci, si = np.cos(inc), np.sin(inc)
    cw, sw = np.cos(argp), np.sin(argp)
    return np.array([
        [cO * cw - sO * sw * ci,  -cO * sw - sO * cw * ci,   sO * si],
        [sO * cw + cO * sw * ci,  -sO * sw + cO * cw * ci,  -cO * si],
        [sw * si,                  cw * si,                  ci    ],
    ])


# ---------------------------------------------------------------------------
# Helpers shared by Lagrange + Keplerian resolvers
# ---------------------------------------------------------------------------


def _scenario_body_by_name(
    scenario: Scenario, name: str, role: str
) -> Body:
    for b in scenario.bodies:
        if b.name == name:
            return b
    raise ScenarioValidationError(
        f"{role} body {name!r} not found in scenario.bodies "
        f"(have: {[b.name for b in scenario.bodies]})"
    )


def _state_for(
    body: Body, epoch: Time, source: EphemerisSource,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve `body` to ICRF barycentric (r, v). Reuses resolve_body's
    EphemerisIc / ExplicitIc paths so Lagrange/Keplerian ICs work
    against any body type the scenario already supports."""
    rb = resolve_body(body, epoch, source)
    return rb.r_km, rb.v_kms


def _mass_kg_for(body: Body) -> float:
    """Mass in kg, falling back to BODY_CONSTANTS lookup if not on the
    Body itself. Lagrange + Keplerian both need this for μ."""
    if body.mass_kg is not None:
        return float(body.mass_kg)
    constant = _lookup_constant(body)
    if constant is None:
        raise ScenarioValidationError(
            f"body {body.name!r}: mass_kg not provided and not in "
            "constants.BODY_CONSTANTS — required for Lagrange/Keplerian IC"
        )
    return float(constant.mass_kg)
