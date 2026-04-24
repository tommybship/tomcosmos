"""REBOUND wrapper: Scenario + ResolvedBody → rebound.Simulation.

Internal units are AU / yr / Msun (gives G ≈ 39.48, numerically well-scaled
for planetary dynamics). The I/O boundary is km / km·s⁻¹ / kg — everything
entering this module gets converted once, and the integration loop converts
back once per sample. This keeps force evaluations away from the mixing-
large-and-small-numbers regime that eats float64 precision.

Unit discipline lives in this module. Nothing that calls us needs to know.
"""
from __future__ import annotations

import rebound
from astropy import units as u

from tomcosmos.state.ic import ResolvedBody, ResolvedTestParticle
from tomcosmos.state.scenario import IntegratorConfig

# Test-particle collision radius: 1 km converted to AU. Only meaningful for
# collision detection; physically they're points.
_TP_COLLISION_RADIUS_KM = 1.0
_TP_COLLISION_RADIUS_AU = float((_TP_COLLISION_RADIUS_KM * u.km).to(u.AU).value)


def build_simulation(
    bodies: list[ResolvedBody],
    test_particles: list[ResolvedTestParticle],
    integrator_config: IntegratorConfig,
) -> rebound.Simulation:
    """Create and configure a `rebound.Simulation` ready to integrate.

    Steps:
      1. Set units to (AU, yr, Msun); never leave REBOUND's default G=1.
      2. Add massive bodies with stable hashes (`hash=<name>`). Indices are
         unstable across `sim.remove()` — every caller should look up by hash.
      3. Add massless test particles with the same hash discipline;
         `N_active` is set to the count of massive bodies and
         `testparticle_type = 0` so they don't perturb the planets
         (important once M5 scales to thousands).
      4. Configure the integrator (integrator name, dt, any per-integrator
         options like `ri_mercurius.r_crit_hill`).
      5. Move to the center-of-mass frame. Without this, the COM drifts
         linearly through ICRF and energy/angular-momentum diagnostics
         pick up spurious linear terms.

    Returns the `Simulation` — caller owns its lifetime.
    """
    if not bodies:
        raise ValueError("build_simulation requires at least one massive body")

    sim = rebound.Simulation()
    sim.units = ("AU", "yr", "Msun")

    for body in bodies:
        r_au = (body.r_km * u.km).to(u.AU).value
        v_au_per_yr = (body.v_kms * u.km / u.s).to(u.AU / u.yr).value
        mass_msun = float((body.mass_kg * u.kg).to(u.Msun).value)
        radius_au = float((body.radius_km * u.km).to(u.AU).value)
        sim.add(
            m=mass_msun,
            r=radius_au,
            x=float(r_au[0]), y=float(r_au[1]), z=float(r_au[2]),
            vx=float(v_au_per_yr[0]),
            vy=float(v_au_per_yr[1]),
            vz=float(v_au_per_yr[2]),
            hash=body.name,
        )

    n_active = len(bodies)

    for particle in test_particles:
        r_au = (particle.r_km * u.km).to(u.AU).value
        v_au_per_yr = (particle.v_kms * u.km / u.s).to(u.AU / u.yr).value
        sim.add(
            m=0.0,
            r=_TP_COLLISION_RADIUS_AU,
            x=float(r_au[0]), y=float(r_au[1]), z=float(r_au[2]),
            vx=float(v_au_per_yr[0]),
            vy=float(v_au_per_yr[1]),
            vz=float(v_au_per_yr[2]),
            hash=particle.name,
        )

    if test_particles:
        sim.N_active = n_active
        sim.testparticle_type = 0

    _configure_integrator(sim, integrator_config)
    sim.move_to_com()
    return sim


def _configure_integrator(sim: rebound.Simulation, cfg: IntegratorConfig) -> None:
    sim.integrator = cfg.name

    if cfg.name in ("whfast", "mercurius"):
        if cfg.timestep is None:  # pragma: no cover — schema rejects this earlier
            raise ValueError(
                f"integrator {cfg.name!r} requires an explicit timestep"
            )
        sim.dt = float(cfg.timestep.to(u.yr).value)

    if cfg.name == "mercurius" and cfg.r_crit_hill is not None:
        sim.ri_mercurius.r_crit_hill = float(cfg.r_crit_hill)
