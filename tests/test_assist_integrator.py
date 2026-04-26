"""End-to-end test for the ASSIST integrator backend.

Validates that `IntegratorConfig.ephemeris_perturbers=True` produces a
working `assist.Extras`-wrapped Simulation that integrates a test
particle through DE440-supplied gravity. The asteroid trajectory is
the load-bearing M5 use case; this test is the architectural proof
that the wiring works before we use it in M5 scenarios.

Marked `assist` so CI can opt out — the kernels are 730 MB total, not
something to download on every run.
"""
from __future__ import annotations

import numpy as np
import pytest
from astropy.time import Time

from tomcosmos.config import assist_asteroid_kernel, assist_planet_kernel
from tomcosmos.state.ic import ResolvedTestParticle
from tomcosmos.state.integrator import build_simulation
from tomcosmos.state.scenario import IntegratorConfig

pytestmark = pytest.mark.assist

AU_KM = 1.495978707e8

# Anchor Mode A tests at J2000 — keeps the existing geometry (probe at
# +1 AU on the x-axis, ~30 km/s tangential) physically meaningful, since
# DE440 has the Sun and barycentric origin defined consistently there.
J2000 = Time("2000-01-01T12:00:00", scale="tdb")


def _kernels_present() -> bool:
    return assist_planet_kernel().exists() and assist_asteroid_kernel().exists()


def _ephemeris_config() -> IntegratorConfig:
    return IntegratorConfig.model_validate({
        "name": "ias15",
        "ephemeris_perturbers": True,
    })


def test_assist_extras_is_attached_to_simulation() -> None:
    """When `ephemeris_perturbers=True`, build_simulation wraps the
    rebound.Simulation with assist.Extras and stashes the bound
    ephem + extras on the sim so they survive the call's frame."""
    if not _kernels_present():
        pytest.skip("ASSIST kernels not present; set TOMCOSMOS_ASSIST_PLANET_KERNEL "
                    "and TOMCOSMOS_ASSIST_ASTEROID_KERNEL or download to data/kernels/")

    probe = ResolvedTestParticle(
        name="probe",
        r_km=np.array([1.5e8, 0.0, 0.0]),  # 1 AU on +x
        v_kms=np.array([0.0, 30.0, 0.0]),  # roughly Earth-ish circular speed
    )
    sim = build_simulation([], [probe], _ephemeris_config(), epoch=J2000)

    assert hasattr(sim, "_tomcosmos_assist_extras"), \
        "Extras not attached — caller would lose ASSIST's force callback"
    assert hasattr(sim, "_tomcosmos_assist_ephem")

    import assist
    assert isinstance(sim._tomcosmos_assist_extras, assist.Extras)
    assert isinstance(sim._tomcosmos_assist_ephem, assist.Ephem)


def test_assist_propagates_test_particle_under_solar_gravity() -> None:
    """Place a test particle on a circular Earth-like orbit and
    propagate one year. Solar gravity from DE440 should bring it
    back near its starting point. This is the simplest end-to-end
    proof that the force loop is firing.

    Note: with ASSIST, sim.units is (AU, day, Msun) — so `sim.integrate`
    arguments are in days, and velocities come back in AU/day."""
    if not _kernels_present():
        pytest.skip("ASSIST kernels not present")

    # Earth-like circular orbit: 1 AU radius, 29.78 km/s tangential.
    r0 = np.array([1.0 * AU_KM, 0.0, 0.0])
    v0 = np.array([0.0, 29.78, 0.0])
    probe = ResolvedTestParticle(name="probe", r_km=r0, v_kms=v0)

    sim = build_simulation([], [probe], _ephemeris_config(), epoch=J2000)
    # sim.t starts at days_past_J2000(J2000) = 0; integrate to t = 365.25
    # = one Julian year in ASSIST's day-units.
    sim.integrate(365.25)

    p = sim.particles[0]
    end_r_km = np.array([p.x, p.y, p.z]) * AU_KM
    # AU/day → km/s
    end_v_kms = np.array([p.vx, p.vy, p.vz]) * (AU_KM / 86400.0)

    # After exactly 1 year, our circular probe should return near its
    # start. Tolerance is generous because we used a textbook 29.78 km/s
    # rather than an exact-period velocity — there's some real period
    # mismatch on top of any integrator drift.
    return_distance_au = float(np.linalg.norm(end_r_km - r0)) / AU_KM
    assert return_distance_au < 0.2, (
        f"Test particle didn't return close to start after 1 yr: "
        f"|Δr| = {return_distance_au:.3f} AU"
    )

    # Speed should remain in the right ballpark (29.78 ± a few km/s).
    end_speed = float(np.linalg.norm(end_v_kms))
    assert 25.0 < end_speed < 35.0, f"Implausible end speed: {end_speed} km/s"


def test_ephemeris_perturbers_rejects_massive_bodies_at_schema_level() -> None:
    """Scenario validator should refuse `ephemeris_perturbers=True`
    alongside any massive body, since that would double-count gravity."""
    from pydantic import ValidationError

    from tomcosmos import Scenario

    with pytest.raises(ValidationError, match="ephemeris_perturbers=True is incompatible"):
        Scenario.model_validate({
            "schema_version": 1, "name": "bad",
            "epoch": "2026-04-23T00:00:00 TDB",
            "duration": "30 day",
            "integrator": {"name": "ias15", "ephemeris_perturbers": True},
            "output": {"format": "parquet", "cadence": "1 day"},
            "bodies": [{"name": "sun", "spice_id": 10, "ic": {"source": "ephemeris"}}],
            "test_particles": [{
                "name": "p", "ic": {"type": "explicit",
                                     "r": [0, 0, 0], "v": [0, 0, 0]},
            }],
        })


def test_no_bodies_allowed_when_ephemeris_perturbers_off() -> None:
    """Without ASSIST, scenarios still need at least one massive body —
    vanilla REBOUND has nothing to integrate against otherwise."""
    from pydantic import ValidationError

    from tomcosmos import Scenario

    with pytest.raises(ValidationError, match="must declare at least one body"):
        Scenario.model_validate({
            "schema_version": 1, "name": "bad",
            "epoch": "2026-04-23T00:00:00 TDB",
            "duration": "30 day",
            "integrator": {"name": "ias15"},
            "output": {"format": "parquet", "cadence": "1 day"},
            "test_particles": [{
                "name": "p", "ic": {"type": "explicit",
                                     "r": [0, 0, 0], "v": [0, 0, 0]},
            }],
        })


def test_assist_rejects_explicit_effects() -> None:
    """ASSIST already includes GR + J2 in its force model. Asking for
    `effects: [gr]` on top is redundant and confusing — refuse it."""
    if not _kernels_present():
        pytest.skip("ASSIST kernels not present")

    cfg = IntegratorConfig.model_validate({
        "name": "ias15",
        "ephemeris_perturbers": True,
        "effects": ["gr"],
    })
    probe = ResolvedTestParticle(
        name="probe",
        r_km=np.array([1.5e8, 0.0, 0.0]),
        v_kms=np.array([0.0, 30.0, 0.0]),
    )
    with pytest.raises(ValueError, match="ASSIST already includes GR"):
        build_simulation([], [probe], cfg, epoch=J2000)


def test_assist_requires_explicit_epoch() -> None:
    """Mode A's force model indexes DE440 by TDB days past J2000, so
    sim.t must be anchored to a real epoch before integration. Building
    a Mode A sim without an epoch is a programmer error — the validator
    should catch it before sim.integrate silently propagates the wrong
    asteroid environment."""
    probe = ResolvedTestParticle(
        name="probe",
        r_km=np.array([1.5e8, 0.0, 0.0]),
        v_kms=np.array([0.0, 30.0, 0.0]),
    )
    with pytest.raises(ValueError, match=r"requires `epoch`"):
        build_simulation([], [probe], _ephemeris_config())
