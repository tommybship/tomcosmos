"""Initial-condition resolution.

Turns declarative `Body` / `TestParticle` scenario entries into concrete
physics inputs: (name, mass_kg, radius_km, r_km, v_kms) all expressed
in ICRF barycentric. The integrator (state/integrator.py) consumes these
directly — it never looks at the YAML-level types.

M1 handles:
    Body.ic = EphemerisIc   → source.query(body_id, epoch)
    Body.ic = ExplicitIc    → frame conversion into ICRF barycentric
    TestParticle.ic = TestParticleExplicitIc → same frame pipeline

M3 extends with:
    TestParticleIc = LagrangeIc | KeplerianIc (via ic_lagrange /
    ic_keplerian helpers that land here).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
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
)


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
        resolve_test_particle(p, scenario.epoch, source)
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
    particle: TestParticle, epoch: Time, source: EphemerisSource
) -> ResolvedTestParticle:
    if isinstance(particle.ic, TestParticleExplicitIc):
        r, v = _to_icrf_barycentric(
            particle.ic.r, particle.ic.v, particle.ic.frame, epoch, source
        )
    else:  # pragma: no cover — M3 adds Lagrange / Keplerian
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
