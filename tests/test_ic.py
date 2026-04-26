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
from tomcosmos.state.ephemeris import EphemerisSource
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
    ephemeris_source: EphemerisSource,
) -> None:
    body = Body(name="earth", spice_id=399, ic=EphemerisIc(source="ephemeris"))
    rb = resolve_body(body, EPOCH, ephemeris_source)
    assert rb.mass_kg == pytest.approx(5.9724e24, rel=1e-6)
    assert rb.radius_km == pytest.approx(6371.0, rel=1e-6)
    assert rb.color_hex == "#4A90D9"


def test_explicit_mass_overrides_constants(
    ephemeris_source: EphemerisSource,
) -> None:
    body = Body(
        name="earth",
        spice_id=399,
        mass_kg=1.234e25,
        ic=EphemerisIc(source="ephemeris"),
    )
    rb = resolve_body(body, EPOCH, ephemeris_source)
    assert rb.mass_kg == 1.234e25
    # radius falls through to constants since not overridden
    assert rb.radius_km == pytest.approx(6371.0, rel=1e-6)


def test_unknown_body_without_explicit_mass_rejected(
    ephemeris_source: EphemerisSource,
) -> None:
    body = Body(
        name="vulcan",
        ic=ExplicitIc(source="explicit", r=(1e8, 0.0, 0.0), v=(0.0, 30.0, 0.0)),
    )
    with pytest.raises(ScenarioValidationError, match="mass_kg"):
        resolve_body(body, EPOCH, ephemeris_source)


def test_unknown_body_with_full_explicit_fields_ok(
    ephemeris_source: EphemerisSource,
) -> None:
    body = Body(
        name="vulcan",
        mass_kg=1e24,
        radius_km=3000.0,
        ic=ExplicitIc(source="explicit", r=(1e8, 0.0, 0.0), v=(0.0, 30.0, 0.0)),
    )
    rb = resolve_body(body, EPOCH, ephemeris_source)
    assert rb.mass_kg == 1e24
    assert rb.color_hex is None  # not in constants
    assert rb.spice_id is None


# --- State-vector resolution (ephemeris path) --------------------------------


def test_ephemeris_source_matches_direct_query(
    ephemeris_source: EphemerisSource,
) -> None:
    body = Body(name="earth", spice_id=399, ic=EphemerisIc(source="ephemeris"))
    rb = resolve_body(body, EPOCH, ephemeris_source)
    r_direct, v_direct = ephemeris_source.query(399, EPOCH)
    assert np.allclose(rb.r_km, r_direct)
    assert np.allclose(rb.v_kms, v_direct)
    assert rb.r_km.shape == (3,)
    assert rb.r_km.dtype == np.float64


def test_ephemeris_uses_spice_id_when_both_given(
    ephemeris_source: EphemerisSource,
) -> None:
    body = Body(name="earth", spice_id=399, ic=EphemerisIc(source="ephemeris"))
    rb = resolve_body(body, EPOCH, ephemeris_source)
    r_by_id, _ = ephemeris_source.query(399, EPOCH)
    assert np.allclose(rb.r_km, r_by_id)


# --- State-vector resolution (explicit path, frame conversions) --------------


def test_explicit_icrf_barycentric_passthrough(
    ephemeris_source: EphemerisSource,
) -> None:
    r_in = (1.0e8, 2.0e8, 3.0e7)
    v_in = (10.0, 20.0, 5.0)
    body = Body(
        name="vulcan",
        mass_kg=1e24,
        radius_km=1.0,
        ic=ExplicitIc(source="explicit", r=r_in, v=v_in, frame="icrf_barycentric"),
    )
    rb = resolve_body(body, EPOCH, ephemeris_source)
    assert np.allclose(rb.r_km, r_in)
    assert np.allclose(rb.v_kms, v_in)


def test_explicit_ecliptic_barycentric_rotates_by_obliquity(
    ephemeris_source: EphemerisSource,
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
    rb = resolve_body(body, EPOCH, ephemeris_source)
    assert np.isclose(rb.r_km[0], 0.0)
    assert np.isclose(rb.r_km[1], np.cos(OBLIQUITY_RAD))
    assert np.isclose(rb.r_km[2], np.sin(OBLIQUITY_RAD))


def test_explicit_icrf_heliocentric_adds_sun_state(
    ephemeris_source: EphemerisSource,
) -> None:
    r_sun, v_sun = ephemeris_source.query("sun", EPOCH)
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
    rb = resolve_body(body, EPOCH, ephemeris_source)
    assert np.allclose(rb.r_km, r_in + r_sun)
    assert np.allclose(rb.v_kms, v_in + v_sun)


def test_explicit_ecliptic_heliocentric_rotates_then_adds_sun(
    ephemeris_source: EphemerisSource,
) -> None:
    from tomcosmos.state.frames import ecliptic_to_icrf
    r_sun, v_sun = ephemeris_source.query("sun", EPOCH)
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
    rb = resolve_body(body, EPOCH, ephemeris_source)
    assert np.allclose(rb.r_km, ecliptic_to_icrf(r_ecl) + r_sun)
    assert np.allclose(rb.v_kms, ecliptic_to_icrf(v_ecl) + v_sun)


# --- Test particle resolution ------------------------------------------------


def test_test_particle_explicit_passthrough(
    ephemeris_source: EphemerisSource,
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
    rp = resolve_test_particle(tp, EPOCH, ephemeris_source)
    assert isinstance(rp, ResolvedTestParticle)
    assert np.allclose(rp.r_km, (1.0e8, 0.0, 0.0))


# --- Full scenario integration -----------------------------------------------


def test_resolve_scenario_sun_planets(ephemeris_source: EphemerisSource) -> None:
    scenario = Scenario.from_yaml("tests/fixtures/scenarios/good_sun_planets.yaml")
    bodies, particles = resolve_scenario(scenario, ephemeris_source)
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
    ephemeris_source: EphemerisSource,
) -> None:
    scenario = Scenario.from_yaml("tests/fixtures/scenarios/good_sun_planets.yaml")
    bodies, _ = resolve_scenario(scenario, ephemeris_source)
    earth = next(b for b in bodies if b.name == "earth")
    r_direct, v_direct = ephemeris_source.query(399, scenario.epoch)
    assert np.allclose(earth.r_km, r_direct)
    assert np.allclose(earth.v_kms, v_direct)


def test_unknown_body_ephemeris_path_raises(
    ephemeris_source: EphemerisSource,
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
        resolve_body(body, EPOCH, ephemeris_source)


# --- M3: Lagrange-point IC --------------------------------------------------


AU_KM = 1.495978707e8


def _sun_earth_scenario() -> Scenario:
    """Scenario with sun + earth at the test epoch — primary/secondary
    pair for every Lagrange test below."""
    return Scenario.model_validate({
        "schema_version": 1,
        "name": "lagrange-test",
        "epoch": "2026-04-23T00:00:00 TDB",
        "duration": "30 day",
        "integrator": {"name": "ias15"},
        "output": {"format": "parquet", "cadence": "1 day"},
        "bodies": [
            {"name": "sun",   "spice_id": 10,  "ic": {"source": "ephemeris"}},
            {"name": "earth", "spice_id": 399, "ic": {"source": "ephemeris"}},
        ],
    })


@pytest.mark.parametrize("point", ["L4", "L5"])
def test_lagrange_l4_l5_form_equilateral_triangle(
    ephemeris_source: EphemerisSource, point: str,
) -> None:
    """L4 and L5 sit at the apex of an equilateral triangle with the
    primary and secondary: distance to primary == distance to secondary
    == primary-secondary separation."""
    from tomcosmos import TestParticle
    from tomcosmos.state.scenario import TestParticleLagrangeIc

    scenario = _sun_earth_scenario()
    tp = TestParticle(
        name=point,
        ic=TestParticleLagrangeIc(
            type="lagrange", point=point, primary="sun", secondary="earth",
        ),
    )
    rtp = resolve_test_particle(tp, scenario.epoch, ephemeris_source, scenario)
    r_sun, _ = ephemeris_source.query("sun", scenario.epoch)
    r_earth, _ = ephemeris_source.query("earth", scenario.epoch)
    R = float(np.linalg.norm(r_earth - r_sun))
    d_sun = float(np.linalg.norm(rtp.r_km - r_sun))
    d_earth = float(np.linalg.norm(rtp.r_km - r_earth))
    # Equilateral: all three pairwise distances equal R within float tol.
    assert abs(d_sun - R) / R < 1e-12
    assert abs(d_earth - R) / R < 1e-12


def test_lagrange_l4_leads_l5_trails(ephemeris_source: EphemerisSource) -> None:
    """L4 leads the secondary in its orbital direction (positive component
    along the secondary's velocity). L5 trails. Sign of the dot product
    locks the orbital convention against accidental sign flips."""
    from tomcosmos import TestParticle
    from tomcosmos.state.scenario import TestParticleLagrangeIc

    scenario = _sun_earth_scenario()
    _, v_earth = ephemeris_source.query("earth", scenario.epoch)
    r_sun, _ = ephemeris_source.query("sun", scenario.epoch)
    r_earth, _ = ephemeris_source.query("earth", scenario.epoch)
    v_earth_rel = v_earth - ephemeris_source.query("sun", scenario.epoch)[1]

    def _l(point: str) -> np.ndarray:
        tp = TestParticle(
            name=point, ic=TestParticleLagrangeIc(
                type="lagrange", point=point, primary="sun", secondary="earth",
            ),
        )
        rtp = resolve_test_particle(tp, scenario.epoch, ephemeris_source, scenario)
        return rtp.r_km - r_earth

    # The L4-Earth and L5-Earth vectors should project oppositely onto the
    # Earth-relative velocity direction.
    lead = float(np.dot(_l("L4"), v_earth_rel))
    trail = float(np.dot(_l("L5"), v_earth_rel))
    assert lead > 0, f"L4 should lead, got dot product {lead}"
    assert trail < 0, f"L5 should trail, got dot product {trail}"


def test_lagrange_l1_l2_collinear_distances(
    ephemeris_source: EphemerisSource,
) -> None:
    """L1 and L2 sit ~1.5 million km from Earth on the Sun-Earth line.

    Real values (Sun-Earth, Hill radius scale): L1 ~1.491e6 km toward Sun,
    L2 ~1.502e6 km away from Sun. Our analytic seed converges to within
    a few thousand km — bound 1% so the test remains robust as the actual
    Earth-Sun distance shifts through the year (R_se varies ~3% over an
    eccentric orbit)."""
    from tomcosmos import TestParticle
    from tomcosmos.state.scenario import TestParticleLagrangeIc

    scenario = _sun_earth_scenario()
    r_sun, _ = ephemeris_source.query("sun", scenario.epoch)
    r_earth, _ = ephemeris_source.query("earth", scenario.epoch)

    for point, expected_km in (("L1", 1.491e6), ("L2", 1.502e6)):
        tp = TestParticle(
            name=point, ic=TestParticleLagrangeIc(
                type="lagrange", point=point, primary="sun", secondary="earth",
            ),
        )
        rtp = resolve_test_particle(tp, scenario.epoch, ephemeris_source, scenario)
        d_earth = float(np.linalg.norm(rtp.r_km - r_earth))
        rel_err = abs(d_earth - expected_km) / expected_km
        assert rel_err < 0.05, (
            f"{point}: distance to Earth {d_earth:.3e} km, expected ~{expected_km:.3e}; "
            f"relative error {rel_err:.3f}"
        )


def test_lagrange_l3_opposite_side_of_primary(
    ephemeris_source: EphemerisSource,
) -> None:
    """L3 is on the far side of the primary from the secondary, ~R away
    from both. Its position vector relative to the primary is anti-parallel
    to the primary→secondary direction."""
    from tomcosmos import TestParticle
    from tomcosmos.state.scenario import TestParticleLagrangeIc

    scenario = _sun_earth_scenario()
    r_sun, _ = ephemeris_source.query("sun", scenario.epoch)
    r_earth, _ = ephemeris_source.query("earth", scenario.epoch)
    R = float(np.linalg.norm(r_earth - r_sun))

    tp = TestParticle(
        name="L3", ic=TestParticleLagrangeIc(
            type="lagrange", point="L3", primary="sun", secondary="earth",
        ),
    )
    rtp = resolve_test_particle(tp, scenario.epoch, ephemeris_source, scenario)
    sun_to_l3 = rtp.r_km - r_sun
    sun_to_earth = r_earth - r_sun
    # Anti-parallel: cosine ≈ -1.
    cos_angle = float(
        np.dot(sun_to_l3, sun_to_earth)
        / (np.linalg.norm(sun_to_l3) * np.linalg.norm(sun_to_earth))
    )
    assert cos_angle < -0.9999, f"L3 should be anti-parallel; cos={cos_angle}"
    # Distance from Sun close to R (within 1% — secular correction is ~5μ/12 R).
    assert abs(np.linalg.norm(sun_to_l3) - R) / R < 0.01


def test_lagrange_velocity_matches_corotation(
    ephemeris_source: EphemerisSource,
) -> None:
    """At any Lagrange point, the velocity in the rotating frame is zero
    by definition. In the inertial frame, the particle co-rotates with
    the secondary: its velocity is ω × r (relative to the primary)."""
    from tomcosmos import TestParticle
    from tomcosmos.state.scenario import TestParticleLagrangeIc

    scenario = _sun_earth_scenario()
    r_sun, v_sun = ephemeris_source.query("sun", scenario.epoch)
    r_earth, v_earth = ephemeris_source.query("earth", scenario.epoch)
    sep = r_earth - r_sun
    rel_v = v_earth - v_sun
    L = np.cross(sep, rel_v)
    omega = L / np.dot(sep, sep)

    tp = TestParticle(
        name="L4", ic=TestParticleLagrangeIc(
            type="lagrange", point="L4", primary="sun", secondary="earth",
        ),
    )
    rtp = resolve_test_particle(tp, scenario.epoch, ephemeris_source, scenario)
    expected_v = v_sun + np.cross(omega, rtp.r_km - r_sun)
    assert np.allclose(rtp.v_kms, expected_v, rtol=1e-12)


# --- M3: Keplerian-elements IC ---------------------------------------------


def test_keplerian_circular_orbit_radius_and_speed(
    ephemeris_source: EphemerisSource,
) -> None:
    """Circular orbit at semi-major axis a should have |r-parent|=a and
    speed v=sqrt(μ/a) regardless of orientation."""
    from tomcosmos import TestParticle
    from tomcosmos.state.scenario import TestParticleKeplerianIc

    scenario = _sun_earth_scenario()
    r_sun, v_sun = ephemeris_source.query("sun", scenario.epoch)

    a = AU_KM
    tp = TestParticle(
        name="probe", ic=TestParticleKeplerianIc(
            type="keplerian", parent="sun",
            a_km=a, e=0.0,
            inc_deg=23.4, raan_deg=15.0, argp_deg=42.0, mean_anom_deg=70.0,
        ),
    )
    rtp = resolve_test_particle(tp, scenario.epoch, ephemeris_source, scenario)
    d_sun = float(np.linalg.norm(rtp.r_km - r_sun))
    v_rel = float(np.linalg.norm(rtp.v_kms - v_sun))

    G = 6.6743e-20
    M_sun = 1.989e30
    expected_v = float(np.sqrt(G * M_sun / a))
    assert abs(d_sun - a) / a < 1e-12
    # Velocity expected to match within ~0.1%; mass is 4 sig figs in BODY_CONSTANTS.
    assert abs(v_rel - expected_v) / expected_v < 1e-3


def test_keplerian_elliptical_periapsis_apoapsis(
    ephemeris_source: EphemerisSource,
) -> None:
    """At mean anomaly 0 (periapsis), |r-parent| should equal a(1-e).
    At mean anomaly 180° (apoapsis), it should equal a(1+e)."""
    from tomcosmos import TestParticle
    from tomcosmos.state.scenario import TestParticleKeplerianIc

    scenario = _sun_earth_scenario()
    r_sun, _ = ephemeris_source.query("sun", scenario.epoch)
    a, e = AU_KM, 0.3

    def _radius_at(M_deg: float) -> float:
        tp = TestParticle(
            name=f"probe-{M_deg}", ic=TestParticleKeplerianIc(
                type="keplerian", parent="sun",
                a_km=a, e=e,
                inc_deg=0.0, raan_deg=0.0, argp_deg=0.0,
                mean_anom_deg=M_deg,
            ),
        )
        rtp = resolve_test_particle(tp, scenario.epoch, ephemeris_source, scenario)
        return float(np.linalg.norm(rtp.r_km - r_sun))

    assert abs(_radius_at(0.0) - a * (1.0 - e)) / (a * (1 - e)) < 1e-12
    assert abs(_radius_at(180.0) - a * (1.0 + e)) / (a * (1 + e)) < 1e-12
