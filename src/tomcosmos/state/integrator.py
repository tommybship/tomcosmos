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

from tomcosmos.state.effects import attach_gr
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
    *,
    assist_kernel_paths: tuple[str, str] | None = None,
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

    When `integrator_config.ephemeris_perturbers=True`, the result is
    wrapped with an ASSIST `Extras` instance that supplies high-precision
    gravity from the Sun, planets, Moon, and 16 asteroid perturbers via
    DE440/sb441-n16 kernels. In this mode, the scenario must contain
    only test particles (the schema enforces zero-bodies), `move_to_com`
    is skipped (ASSIST works in the SSB frame defined by its kernels),
    and the `effects` list is rejected as redundant (ASSIST already
    includes GR + J2 internally).

    `assist_kernel_paths` defaults to (config.assist_planet_kernel(),
    config.assist_asteroid_kernel()) when None and ASSIST is in use.

    Returns the `Simulation` — caller owns its lifetime.
    """
    if integrator_config.ephemeris_perturbers:
        if not test_particles:
            raise ValueError(
                "ephemeris_perturbers=True requires at least one test particle "
                "to integrate (the major bodies come from ASSIST's kernels)"
            )
        if bodies:
            raise ValueError(
                "ephemeris_perturbers=True is incompatible with massive bodies; "
                "the schema should have caught this — check Scenario validation"
            )
        if integrator_config.effects:
            raise ValueError(
                "ephemeris_perturbers=True ignores integrator.effects "
                f"({integrator_config.effects!r}); ASSIST already includes "
                "GR and J2 in its force model. Drop the effects list."
            )
    elif not bodies:
        raise ValueError("build_simulation requires at least one massive body")

    sim = rebound.Simulation()
    # ASSIST's force loop uses AU / day / Msun internally — its kernel
    # times are TDB seconds past J2000, which it converts to days for
    # `sim.t`. Setting yr-units would silently misinterpret velocities
    # (a probe moving 30 km/s ≈ 6 AU/yr would be read as 6 AU/day,
    # which is 365× too fast). Vanilla REBOUND scenarios stay on
    # yr-units — that's been our convention since M0.
    if integrator_config.ephemeris_perturbers:
        sim.units = ("AU", "day", "Msun")
    else:
        sim.units = ("AU", "yr", "Msun")

    # Velocity-conversion target depends on the time-unit branch above.
    v_unit = u.AU / u.day if integrator_config.ephemeris_perturbers else u.AU / u.yr

    for body in bodies:
        r_au = (body.r_km * u.km).to(u.AU).value
        v_au_per_t = (body.v_kms * u.km / u.s).to(v_unit).value
        mass_msun = float((body.mass_kg * u.kg).to(u.Msun).value)
        radius_au = float((body.radius_km * u.km).to(u.AU).value)
        sim.add(
            m=mass_msun,
            r=radius_au,
            x=float(r_au[0]), y=float(r_au[1]), z=float(r_au[2]),
            vx=float(v_au_per_t[0]),
            vy=float(v_au_per_t[1]),
            vz=float(v_au_per_t[2]),
            hash=body.name,
        )

    n_active = len(bodies)

    for particle in test_particles:
        r_au = (particle.r_km * u.km).to(u.AU).value
        v_au_per_t = (particle.v_kms * u.km / u.s).to(v_unit).value
        sim.add(
            m=0.0,
            r=_TP_COLLISION_RADIUS_AU,
            x=float(r_au[0]), y=float(r_au[1]), z=float(r_au[2]),
            vx=float(v_au_per_t[0]),
            vy=float(v_au_per_t[1]),
            vz=float(v_au_per_t[2]),
            hash=particle.name,
        )

    if test_particles:
        sim.N_active = n_active
        sim.testparticle_type = 0

    _configure_integrator(sim, integrator_config)

    if integrator_config.ephemeris_perturbers:
        _attach_assist(sim, assist_kernel_paths)
        # Skip move_to_com: ASSIST's force model is defined in the SSB
        # frame implied by its DE440 kernel. Shifting the test particles
        # to their own COM would put them in a non-inertial frame that
        # ASSIST's hardcoded perturber positions don't share.
    else:
        _apply_effects(sim, integrator_config)
        sim.move_to_com()
    return sim


def _attach_assist(
    sim: rebound.Simulation,
    kernel_paths: tuple[str, str] | None,
) -> None:
    """Wrap the Simulation with `assist.Extras` so the force loop pulls
    gravity from DE440 + sb441-n16 instead of from in-sim massive
    bodies. Stash the Extras and Ephem on the sim as hidden attributes
    — REBOUND's C side holds only callback pointers, so the Python
    objects must outlive any integration call."""
    import assist  # local import keeps the optional dep out of cold paths
    if kernel_paths is None:
        from tomcosmos.config import assist_asteroid_kernel, assist_planet_kernel
        kernel_paths = (str(assist_planet_kernel()), str(assist_asteroid_kernel()))
    planet_kernel, asteroid_kernel = kernel_paths
    ephem = assist.Ephem(planet_kernel, asteroid_kernel)
    extras = assist.Extras(sim, ephem)
    # Keep the Python objects alive for the lifetime of the simulation.
    sim._tomcosmos_assist_ephem = ephem
    sim._tomcosmos_assist_extras = extras


def _configure_integrator(sim: rebound.Simulation, cfg: IntegratorConfig) -> None:
    sim.integrator = cfg.name

    if cfg.name in ("whfast", "mercurius"):
        if cfg.timestep is None:  # pragma: no cover — schema rejects this earlier
            raise ValueError(
                f"integrator {cfg.name!r} requires an explicit timestep"
            )
        # Match the time unit chosen in build_simulation: days when
        # ASSIST is on, years otherwise.
        time_unit = u.day if cfg.ephemeris_perturbers else u.yr
        sim.dt = float(cfg.timestep.to(time_unit).value)

    if cfg.name == "mercurius" and cfg.r_crit_hill is not None:
        sim.ri_mercurius.r_crit_hill = float(cfg.r_crit_hill)


def _apply_effects(sim: rebound.Simulation, cfg: IntegratorConfig) -> None:
    """Attach optional physics effects (GR, etc.) after the particles are in.

    Callbacks are stashed on the sim as a hidden attribute so they stay
    alive for as long as the simulation does — REBOUND's C side holds only
    a CFUNCTYPE pointer.
    """
    if not cfg.effects:
        return
    callbacks: list[object] = []
    if "gr" in cfg.effects:
        callbacks.append(attach_gr(sim, sun_hash=rebound.hash("sun").value))
    sim._tomcosmos_effect_callbacks = callbacks
