"""IC resolution tests.

Marked @pytest.mark.ephemeris because full resolution needs the DE440s
kernel loaded. Pure-frame-conversion tests could run without it, but
most of the interesting cases query the ephemeris somewhere.
"""
from __future__ import annotations

import numpy as np
import pytest
from astropy.time import Time

from tomcosmos import Body, EphemerisIc, ExplicitIc, Scenario
from tomcosmos.exceptions import ScenarioValidationError, UnknownBodyError
from tomcosmos.state.ephemeris import SkyfieldSource
from tomcosmos.state.frames import OBLIQUITY_RAD
from tomcosmos.state.ic import (
    ResolvedBody,
    ResolvedTestParticle,
    resolve_body,
    resolve_scenario,
    resolve_test_particle,
)

pytestmark = pytest.mark.ephemeris

EPOCH = Time("2026-04-23T00:00:00", scale="tdb")


# --- Body mass/radius resolution ---------------------------------------------


def test_mass_radius_from_constants_when_omitted(
    skyfield_source: SkyfieldSource,
) -> None:
    body = Body(name="earth", spice_id=399, ic=EphemerisIc(source="ephemeris"))
    rb = resolve_body(body, EPOCH, skyfield_source)
    assert rb.mass_kg == pytest.approx(5.9724e24, rel=1e-6)
    assert rb.radius_km == pytest.approx(6371.0, rel=1e-6)
    assert rb.color_hex == "#4A90D9"


def test_explicit_mass_overrides_constants(
    skyfield_source: SkyfieldSource,
) -> None:
    body = Body(
        name="earth",
        spice_id=399,
        mass_kg=1.234e25,
        ic=EphemerisIc(source="ephemeris"),
    )
    rb = resolve_body(body, EPOCH, skyfield_source)
    assert rb.mass_kg == 1.234e25
    # radius falls through to constants since not overridden
    assert rb.radius_km == pytest.approx(6371.0, rel=1e-6)


def test_unknown_body_without_explicit_mass_rejected(
    skyfield_source: SkyfieldSource,
) -> None:
    body = Body(
        name="vulcan",
        ic=ExplicitIc(source="explicit", r=(1e8, 0.0, 0.0), v=(0.0, 30.0, 0.0)),
    )
    with pytest.raises(ScenarioValidationError, match="mass_kg"):
        resolve_body(body, EPOCH, skyfield_source)


def test_unknown_body_with_full_explicit_fields_ok(
    skyfield_source: SkyfieldSource,
) -> None:
    body = Body(
        name="vulcan",
        mass_kg=1e24,
        radius_km=3000.0,
        ic=ExplicitIc(source="explicit", r=(1e8, 0.0, 0.0), v=(0.0, 30.0, 0.0)),
    )
    rb = resolve_body(body, EPOCH, skyfield_source)
    assert rb.mass_kg == 1e24
    assert rb.color_hex is None  # not in constants
    assert rb.spice_id is None


# --- State-vector resolution (ephemeris path) --------------------------------


def test_ephemeris_source_matches_direct_query(
    skyfield_source: SkyfieldSource,
) -> None:
    body = Body(name="earth", spice_id=399, ic=EphemerisIc(source="ephemeris"))
    rb = resolve_body(body, EPOCH, skyfield_source)
    r_direct, v_direct = skyfield_source.query(399, EPOCH)
    assert np.allclose(rb.r_km, r_direct)
    assert np.allclose(rb.v_kms, v_direct)
    assert rb.r_km.shape == (3,)
    assert rb.r_km.dtype == np.float64


def test_ephemeris_uses_spice_id_when_both_given(
    skyfield_source: SkyfieldSource,
) -> None:
    body = Body(name="earth", spice_id=399, ic=EphemerisIc(source="ephemeris"))
    rb = resolve_body(body, EPOCH, skyfield_source)
    r_by_id, _ = skyfield_source.query(399, EPOCH)
    assert np.allclose(rb.r_km, r_by_id)


# --- State-vector resolution (explicit path, frame conversions) --------------


def test_explicit_icrf_barycentric_passthrough(
    skyfield_source: SkyfieldSource,
) -> None:
    r_in = (1.0e8, 2.0e8, 3.0e7)
    v_in = (10.0, 20.0, 5.0)
    body = Body(
        name="vulcan",
        mass_kg=1e24,
        radius_km=1.0,
        ic=ExplicitIc(source="explicit", r=r_in, v=v_in, frame="icrf_barycentric"),
    )
    rb = resolve_body(body, EPOCH, skyfield_source)
    assert np.allclose(rb.r_km, r_in)
    assert np.allclose(rb.v_kms, v_in)


def test_explicit_ecliptic_barycentric_rotates_by_obliquity(
    skyfield_source: SkyfieldSource,
) -> None:
    # Ecliptic +Y → ICRF (0, cos eps, sin eps)
    body = Body(
        name="vulcan",
        mass_kg=1e24,
        radius_km=1.0,
        ic=ExplicitIc(
            source="explicit",
            r=(0.0, 1.0, 0.0),
            v=(0.0, 0.0, 0.0),
            frame="ecliptic_j2000_barycentric",
        ),
    )
    rb = resolve_body(body, EPOCH, skyfield_source)
    assert np.isclose(rb.r_km[0], 0.0)
    assert np.isclose(rb.r_km[1], np.cos(OBLIQUITY_RAD))
    assert np.isclose(rb.r_km[2], np.sin(OBLIQUITY_RAD))


def test_explicit_icrf_heliocentric_adds_sun_state(
    skyfield_source: SkyfieldSource,
) -> None:
    r_sun, v_sun = skyfield_source.query("sun", EPOCH)
    r_in = np.array([1.0e8, 2.0e8, 3.0e7])
    v_in = np.array([10.0, 20.0, 5.0])
    body = Body(
        name="vulcan",
        mass_kg=1e24,
        radius_km=1.0,
        ic=ExplicitIc(
            source="explicit",
            r=tuple(r_in),
            v=tuple(v_in),
            frame="icrf_heliocentric",
        ),
    )
    rb = resolve_body(body, EPOCH, skyfield_source)
    assert np.allclose(rb.r_km, r_in + r_sun)
    assert np.allclose(rb.v_kms, v_in + v_sun)


def test_explicit_ecliptic_heliocentric_rotates_then_adds_sun(
    skyfield_source: SkyfieldSource,
) -> None:
    from tomcosmos.state.frames import ecliptic_to_icrf
    r_sun, v_sun = skyfield_source.query("sun", EPOCH)
    r_ecl = np.array([1.0e8, 0.0, 0.0])
    v_ecl = np.array([0.0, 30.0, 0.0])
    body = Body(
        name="vulcan",
        mass_kg=1e24,
        radius_km=1.0,
        ic=ExplicitIc(
            source="explicit",
            r=tuple(r_ecl),
            v=tuple(v_ecl),
            frame="ecliptic_j2000_heliocentric",
        ),
    )
    rb = resolve_body(body, EPOCH, skyfield_source)
    assert np.allclose(rb.r_km, ecliptic_to_icrf(r_ecl) + r_sun)
    assert np.allclose(rb.v_kms, ecliptic_to_icrf(v_ecl) + v_sun)


# --- Test particle resolution ------------------------------------------------


def test_test_particle_explicit_passthrough(
    skyfield_source: SkyfieldSource,
) -> None:
    from tomcosmos import TestParticle, TestParticleExplicitIc
    tp = TestParticle(
        name="probe-1",
        ic=TestParticleExplicitIc(
            type="explicit",
            r=(1.0e8, 0.0, 0.0),
            v=(0.0, 30.0, 0.0),
        ),
    )
    rp = resolve_test_particle(tp, EPOCH, skyfield_source)
    assert isinstance(rp, ResolvedTestParticle)
    assert np.allclose(rp.r_km, (1.0e8, 0.0, 0.0))


# --- Full scenario integration -----------------------------------------------


def test_resolve_scenario_sun_planets(skyfield_source: SkyfieldSource) -> None:
    scenario = Scenario.from_yaml("tests/fixtures/scenarios/good_sun_planets.yaml")
    bodies, particles = resolve_scenario(scenario, skyfield_source)
    assert len(bodies) == 9
    assert particles == []
    names = [b.name for b in bodies]
    assert names == [
        "sun", "mercury", "venus", "earth", "mars",
        "jupiter", "saturn", "uranus", "neptune",
    ]
    for rb in bodies:
        assert isinstance(rb, ResolvedBody)
        assert rb.r_km.shape == (3,)
        assert rb.v_kms.shape == (3,)
        assert rb.mass_kg > 0
        assert rb.radius_km > 0


def test_resolve_scenario_preserves_ephemeris_positions(
    skyfield_source: SkyfieldSource,
) -> None:
    scenario = Scenario.from_yaml("tests/fixtures/scenarios/good_sun_planets.yaml")
    bodies, _ = resolve_scenario(scenario, skyfield_source)
    earth = next(b for b in bodies if b.name == "earth")
    r_direct, v_direct = skyfield_source.query(399, scenario.epoch)
    assert np.allclose(earth.r_km, r_direct)
    assert np.allclose(earth.v_kms, v_direct)


def test_unknown_body_ephemeris_path_raises(
    skyfield_source: SkyfieldSource,
) -> None:
    # Pluto isn't in BODY_CONSTANTS (M1 stops at Neptune). Providing mass/radius
    # bypasses the constants-lookup check so we reach the ephemeris query,
    # which is where UnknownBodyError fires.
    body = Body(
        name="pluto",
        mass_kg=1.303e22,
        radius_km=1188.3,
        ic=EphemerisIc(source="ephemeris"),
    )
    with pytest.raises(UnknownBodyError):
        resolve_body(body, EPOCH, skyfield_source)
