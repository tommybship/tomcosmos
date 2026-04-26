"""REBOUND wrapper: Scenario + ResolvedBody → rebound.Simulation.

Two modes — picked by `integrator.ephemeris_perturbers`:

  - **Mode A** (`ephemeris_perturbers=True`): wrap the simulation with
    `assist.Extras` so the force loop pulls gravity from DE440 / sb441-n16
    directly. Real solar system at JPL precision; only test particles
    are integrated (the schema validator enforces zero massive bodies).
    Internal units are AU / day / Msun to match ASSIST's force model.
    Used for accurate small-body / asteroid / mission propagation
    against the real ephemeris.

  - **Mode B** (`ephemeris_perturbers=False`): vanilla REBOUND, every
    massive body declared explicitly in the scenario. Optional REBOUNDx
    forces (currently `gr`) attach via `state.effects`. Internal units
    are AU / yr / Msun. Used for counterfactual scenarios — Lagrange
    demos, hypothetical systems, "what if Planet 9 existed."

The I/O boundary is km / km·s⁻¹ / kg in both modes — everything entering
this module gets converted once, and the integration loop converts back
once per sample. This keeps force evaluations away from the mixing-
large-and-small-numbers regime that eats float64 precision.

Unit discipline lives in this module. Nothing that calls us needs to know.
"""
from __future__ import annotations

import rebound
from astropy import units as u
from astropy.time import Time

from tomcosmos.state.effects import attach_gr
from tomcosmos.state.ic import ResolvedBody, ResolvedTestParticle
from tomcosmos.state.scenario import IntegratorConfig
from tomcosmos.state.sim_units import days_past_j2000

# Test-particle collision radius: 1 km converted to AU. Only meaningful for
# collision detection; physically they're points.
_TP_COLLISION_RADIUS_KM = 1.0
_TP_COLLISION_RADIUS_AU = float((_TP_COLLISION_RADIUS_KM * u.km).to(u.AU).value)


def build_simulation(
    bodies: list[ResolvedBody],
    test_particles: list[ResolvedTestParticle],
    integrator_config: IntegratorConfig,
    *,
    epoch: Time | None = None,
    assist_kernel_paths: tuple[str, str] | None = None,
) -> rebound.Simulation:
    """Create and configure a `rebound.Simulation` ready to integrate.

    Steps:
      1. Set units to (AU, yr, Msun) for Mode B or (AU, day, Msun) for
         Mode A. Never leave REBOUND's default G=1.
      2. Add massive bodies with stable hashes (`hash=<name>`). Indices are
         unstable across `sim.remove()` — every caller should look up by hash.
      3. Add massless test particles with the same hash discipline;
         `N_active` is set to the count of massive bodies and
         `testparticle_type = 0` so they don't perturb the planets
         (important once M5 scales to thousands).
      4. Configure the integrator (integrator name, dt, any per-integrator
         options like `ri_mercurius.r_crit_hill`).
      5. In Mode A, anchor `sim.t` to TDB days past J2000 for the supplied
         `epoch` so ASSIST's kernel lookups index the right ephemeris
         instant. Caller passes `epoch=scenario.epoch`.
      6. Move to the center-of-mass frame (Mode B only). Without this, the
         COM drifts linearly through ICRF and energy/angular-momentum
         diagnostics pick up spurious linear terms. Mode A skips this:
         ASSIST's force model is defined in the SSB frame implied by
         DE440 — shifting the test particles to their own COM would put
         them in a non-inertial frame ASSIST's hardcoded perturber
         positions don't share.

    When `integrator_config.ephemeris_perturbers=True`, the result is
    wrapped with an ASSIST `Extras` instance that supplies high-precision
    gravity from the Sun, planets, Moon, and 16 asteroid perturbers via
    DE440/sb441-n16 kernels. In this mode, the scenario must contain
    only test particles (the schema enforces zero-bodies), and the
    `effects` list is rejected as redundant (ASSIST already includes
    GR + J2 internally).

    `epoch` is **required** in Mode A — without it, ASSIST treats the
    scenario as starting at J2000 regardless of when the user actually
    wanted, which silently lies about the asteroid's environment by
    decades. Mode B accepts `epoch=None` (the default); sim.t starts
    at 0 and represents "elapsed time from start" rather than an
    absolute ephemeris instant.

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
        if epoch is None:
            raise ValueError(
                "ephemeris_perturbers=True requires `epoch` so sim.t can be "
                "anchored to TDB days past J2000 for ASSIST's kernel lookups."
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
        # Anchor sim.t before attaching Extras. ASSIST reads sim.t as TDB
        # days past J2000 to index the DE440 / sb441-n16 kernels; leaving
        # it at 0 would make the asteroid feel J2000's planet positions
        # regardless of the scenario's actual epoch.
        # `epoch` is non-None here per the validation block above; mypy can't
        # see across the early-raise so we narrow it explicitly.
        assert epoch is not None
        sim.t = days_past_j2000(epoch)
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

    Mode B only — Mode A's ASSIST force model already includes GR + J2,
    and the schema validator rejects an `effects` list when ASSIST is on.

    Handles are stashed on the sim as a hidden attribute so they stay
    alive for as long as the simulation does — REBOUND's C side holds
    only a callback pointer.
    """
    if not cfg.effects:
        return
    handles: list[object] = []
    if "gr" in cfg.effects:
        handles.append(attach_gr(sim))
    sim._tomcosmos_effect_callbacks = handles
