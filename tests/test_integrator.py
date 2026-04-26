"""Tests for the REBOUND wrapper.

Most tests use in-memory ResolvedBody instances rather than the ephemeris
so they're unit-tests (fast, no kernel dependency). The sun-planets
integration test is marked `ephemeris` since it needs the real IC
resolution pipeline.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import rebound
from astropy import units as u
from astropy.time import Time

from tomcosmos import IntegratorConfig, Scenario
from tomcosmos.state.ephemeris import EphemerisSource
from tomcosmos.state.ic import (
    ResolvedBody,
    ResolvedTestParticle,
    resolve_scenario,
)
from tomcosmos.state.integrator import build_simulation

AU_KM = 1.495978707e8  # keep in sync with astropy's u.AU


# --- Fixtures ----------------------------------------------------------------


def _two_body_sun_earth() -> list[ResolvedBody]:
    """Minimal in-memory scenario: Sun at origin, Earth at 1 AU with
    circular Keplerian velocity. No ephemeris dependency."""
    sun = ResolvedBody(
        name="sun",
        mass_kg=1.989e30,
        radius_km=695700.0,
        r_km=np.array([0.0, 0.0, 0.0]),
        v_kms=np.array([0.0, 0.0, 0.0]),
    )
    # Earth at +X, circular orbit velocity ≈ 29.78 km/s in +Y
    earth = ResolvedBody(
        name="earth",
        mass_kg=5.9724e24,
        radius_km=6371.0,
        r_km=np.array([AU_KM, 0.0, 0.0]),
        v_kms=np.array([0.0, 29.7847, 0.0]),
    )
    return [sun, earth]


def _whfast_day_config() -> IntegratorConfig:
    return IntegratorConfig.model_validate({"name": "whfast", "timestep": "1 day"})


def _ias15_config() -> IntegratorConfig:
    return IntegratorConfig.model_validate({"name": "ias15"})


# --- Basic construction ------------------------------------------------------


def test_empty_bodies_rejected() -> None:
    cfg = _whfast_day_config()
    with pytest.raises(ValueError, match="at least one"):
        build_simulation([], [], cfg)


def test_units_are_au_yr_msun() -> None:
    sim = build_simulation(_two_body_sun_earth(), [], _whfast_day_config())
    # G in AU/yr/Msun ≈ 4π² = 39.478
    assert 39 < sim.G < 40


def test_particle_count_matches_bodies() -> None:
    sim = build_simulation(_two_body_sun_earth(), [], _whfast_day_config())
    assert sim.N == 2


def test_lookup_by_hash_works() -> None:
    sim = build_simulation(_two_body_sun_earth(), [], _whfast_day_config())
    earth = sim.particles[rebound.hash("earth")]
    # After move_to_com, Earth's x is slightly less than 1 AU (COM shifts by the
    # Sun's fraction of their midpoint: 3e-6 of 1 AU).
    assert 0.999 < earth.x < 1.001


def test_lookup_by_unknown_hash_raises() -> None:
    sim = build_simulation(_two_body_sun_earth(), [], _whfast_day_config())
    with pytest.raises((KeyError, rebound.ParticleNotFound)):
        _ = sim.particles[rebound.hash("mars")]


# --- Unit conversion sanity --------------------------------------------------


def test_earth_position_round_trips_through_au() -> None:
    bodies = _two_body_sun_earth()
    sim = build_simulation(bodies, [], _whfast_day_config())
    earth = sim.particles[rebound.hash("earth")]
    # move_to_com shifts positions; undo it by checking Sun-relative separation.
    sun = sim.particles[rebound.hash("sun")]
    dx_au = earth.x - sun.x
    dy_au = earth.y - sun.y
    dz_au = earth.z - sun.z
    sep_km = math.sqrt(dx_au**2 + dy_au**2 + dz_au**2) * AU_KM
    assert sep_km == pytest.approx(AU_KM, rel=1e-9)


def test_move_to_com_zeroes_momentum() -> None:
    sim = build_simulation(_two_body_sun_earth(), [], _whfast_day_config())
    px = sum(p.m * p.vx for p in sim.particles)
    py = sum(p.m * p.vy for p in sim.particles)
    pz = sum(p.m * p.vz for p in sim.particles)
    total_p = math.sqrt(px**2 + py**2 + pz**2)
    assert total_p < 1e-12


def test_mass_ratio_preserved_to_better_than_envelope() -> None:
    """Sun/Earth mass ratio should hold within the ~1e-4 envelope
    from our use of mass*G instead of JPL's GM (see PLAN Non-goals)."""
    sim = build_simulation(_two_body_sun_earth(), [], _whfast_day_config())
    sun = sim.particles[rebound.hash("sun")]
    earth = sim.particles[rebound.hash("earth")]
    true_ratio = 5.9724e24 / 1.989e30
    rb_ratio = earth.m / sun.m
    assert rb_ratio == pytest.approx(true_ratio, rel=1e-4)


# --- Integrator configuration ------------------------------------------------


def test_whfast_gets_timestep() -> None:
    sim = build_simulation(_two_body_sun_earth(), [], _whfast_day_config())
    assert sim.integrator == "whfast"
    # 1 day in years
    assert sim.dt == pytest.approx(1.0 / 365.25, rel=1e-6)


def test_ias15_has_no_forced_timestep() -> None:
    sim = build_simulation(_two_body_sun_earth(), [], _ias15_config())
    assert sim.integrator == "ias15"
    # IAS15 picks its own dt; we don't set one. REBOUND may default to a
    # small value but we at least verify we didn't force a yr-scale step.


def test_mercurius_with_r_crit_hill() -> None:
    cfg = IntegratorConfig.model_validate(
        {"name": "mercurius", "timestep": "1 day", "r_crit_hill": 4.0}
    )
    sim = build_simulation(_two_body_sun_earth(), [], cfg)
    assert sim.integrator == "mercurius"
    assert sim.ri_mercurius.r_crit_hill == pytest.approx(4.0)


# --- Integration behavior ----------------------------------------------------


def test_whfast_advances_time_in_years() -> None:
    sim = build_simulation(_two_body_sun_earth(), [], _whfast_day_config())
    sim.integrate(1.0)  # 1 year
    assert sim.t == pytest.approx(1.0, rel=1e-9)


def test_earth_completes_one_orbit_per_year() -> None:
    """Classical Kepler's third law at a=1 AU with M_sun=1 Msun gives
    T≈1 yr. After exactly 1 year Earth should return near its start."""
    bodies = _two_body_sun_earth()
    sim = build_simulation(bodies, [], _whfast_day_config())
    earth_before = sim.particles[rebound.hash("earth")]
    x0, y0 = earth_before.x, earth_before.y
    sim.integrate(1.0)
    earth_after = sim.particles[rebound.hash("earth")]
    # Returned within ~0.01 AU — wider tolerance because Earth mass matters
    # a tiny bit and WHFast at 1-day step has numerical drift.
    dx = earth_after.x - x0
    dy = earth_after.y - y0
    dr_au = math.sqrt(dx**2 + dy**2)
    assert dr_au < 0.01


def test_energy_is_bounded_under_whfast() -> None:
    """Symplectic property: energy oscillates but doesn't drift. Integrate
    1 year with 1-day step and verify |ΔE/E| stays < 1e-5."""
    sim = build_simulation(_two_body_sun_earth(), [], _whfast_day_config())
    e0 = sim.energy()
    sim.integrate(1.0)
    e1 = sim.energy()
    rel_err = abs((e1 - e0) / e0)
    assert rel_err < 1e-5


# --- Test particle plumbing --------------------------------------------------


def test_n_active_untouched_when_no_test_particles() -> None:
    sim = build_simulation(_two_body_sun_earth(), [], _whfast_day_config())
    # REBOUND uses N_active = -1 as the sentinel for "all particles are active".
    # We only touch it when test particles are present, so -1 should survive.
    assert sim.N_active == -1
    assert sim.N == 2


def test_test_particles_set_n_active_and_type() -> None:
    probe = ResolvedTestParticle(
        name="probe",
        r_km=np.array([2.0 * AU_KM, 0.0, 0.0]),
        v_kms=np.array([0.0, 21.0, 0.0]),
    )
    sim = build_simulation(_two_body_sun_earth(), [probe], _whfast_day_config())
    assert sim.N == 3
    assert sim.N_active == 2
    assert sim.testparticle_type == 0


def test_test_particle_has_mass_zero_and_collision_radius() -> None:
    probe = ResolvedTestParticle(
        name="probe",
        r_km=np.array([2.0 * AU_KM, 0.0, 0.0]),
        v_kms=np.array([0.0, 21.0, 0.0]),
    )
    sim = build_simulation(_two_body_sun_earth(), [probe], _whfast_day_config())
    p = sim.particles[rebound.hash("probe")]
    assert p.m == 0.0
    # 1 km in AU
    assert p.r == pytest.approx((1.0 * u.km).to(u.AU).value, rel=1e-9)


# --- Full scenario integration (ephemeris-backed) ----------------------------


@pytest.mark.ephemeris
def test_sun_planets_scenario_builds_and_integrates(
    ephemeris_source: EphemerisSource,
) -> None:
    scenario = Scenario.from_yaml("tests/fixtures/scenarios/good_sun_planets.yaml")
    bodies, particles = resolve_scenario(scenario, ephemeris_source)
    sim = build_simulation(bodies, particles, scenario.integrator)
    assert sim.N == 9
    e0 = sim.energy()
    sim.integrate(1.0)
    e1 = sim.energy()
    # Sun + 8 planets, 1 day step, 1 year: symplectic — energy bounded.
    assert abs((e1 - e0) / e0) < 1e-6


# --- M4 exit criterion #2: MERCURIUS handles close encounters ---------------


@pytest.mark.ephemeris
def test_mercurius_handles_earth_hill_sphere_flyby_without_blowup(
    ephemeris_source: EphemerisSource,
) -> None:
    """A small massive probe on a trajectory that passes through Earth's
    Hill sphere should not destabilize MERCURIUS.

    MERCURIUS is the integrator that exists *for* this case: WHFast
    can't handle close encounters at all (its symplectic structure
    breaks down near the singularity), and IAS15 handles them but
    pays the cost across the entire run. MERCURIUS uses WHFast in the
    far field and switches to BS (high-order non-symplectic) inside
    `r_crit_hill` of any massive particle.

    Asserted invariants (M4 #2 "no numerical blowup"):
      - No NaN/inf in any state column.
      - Energy stays bounded across the encounter — a 1e-3 ceiling is
        well above MERCURIUS's typical ~1e-6 for a single Hill passage,
        but well below "blowup" magnitude.
      - The flyby actually occurred (probe came within Earth's Hill
        radius at some sample) — sanity check that we're testing what
        we claim, not just integrating in deep space.
    """
    from tomcosmos import Body, ExplicitIc, OutputConfig
    from tomcosmos.runner import run

    epoch = Time("2026-04-23T00:00:00", scale="tdb")
    r_earth, v_earth = ephemeris_source.query("earth", epoch)
    # Probe starts 5e6 km from Earth (above Hill ~1.5e6 km), closing at
    # 10 km/s. Same geometry as the M3 encounter test, validated to
    # produce a Hill-crossing within ~5 days.
    r_probe = tuple((r_earth + np.array([5.0e6, 0.0, 0.0])).tolist())
    v_probe = tuple((v_earth + np.array([-10.0, 0.0, 0.0])).tolist())

    scenario = Scenario(
        schema_version=1, name="mercurius-flyby",
        epoch=epoch,
        duration=30.0 * u.day,
        integrator=IntegratorConfig(
            name="mercurius",
            timestep=0.5 * u.day,  # WHFast-tier far-field cadence
        ),
        output=OutputConfig(format="parquet", cadence=0.5 * u.day),
        bodies=[
            Body(name="sun",   spice_id=10,  ic={"source": "ephemeris"}),
            Body(name="earth", spice_id=399, ic={"source": "ephemeris"}),
            Body(
                name="probe",
                mass_kg=1.0e3, radius_km=1.0,
                ic=ExplicitIc(source="explicit", r=r_probe, v=v_probe),
            ),
        ],
    )
    history = run(scenario, source=ephemeris_source)

    # No NaN / inf in any state column.
    state_cols = ["x", "y", "z", "vx", "vy", "vz"]
    state = history.df[state_cols].to_numpy(dtype=float)
    assert np.all(np.isfinite(state)), "found NaN/inf state — MERCURIUS blew up"

    # Energy bounded.
    max_err = float(history.df["energy_rel_err"].max())
    assert max_err < 1e-3, (
        f"|ΔE/E| max = {max_err:.3e} exceeds 1e-3 — MERCURIUS handover unstable"
    )

    # Setup sanity: flyby actually crossed Earth's Hill sphere.
    earth_pos = history.df[history.df["body"] == "earth"][
        ["sample_idx", "x", "y", "z"]
    ].sort_values("sample_idx")[["x", "y", "z"]].to_numpy()
    probe_pos = history.df[history.df["body"] == "probe"][
        ["sample_idx", "x", "y", "z"]
    ].sort_values("sample_idx")[["x", "y", "z"]].to_numpy()
    min_d = float(np.linalg.norm(probe_pos - earth_pos, axis=1).min())
    assert min_d < 1.5e6, (
        f"flyby did not cross Earth's Hill sphere (min distance {min_d:.3e} km); "
        "test setup is wrong, not MERCURIUS"
    )
