"""Unit-conversion helpers driven by `rebound.Simulation.units`.

REBOUND's internal time + length + mass units are settable per-sim via
`sim.units = (length, time, mass)`. tomcosmos picks the unit triple at
`build_simulation` time based on which integration mode is active:

  - Mode A (`integrator.ephemeris_perturbers=True`): ``("AU", "day", "Msun")``
    to match ASSIST's force loop, which expects times in TDB days past
    J2000 to index DE440 / sb441-n16.
  - Mode B (`ephemeris_perturbers=False`): ``("AU", "yr", "Msun")``,
    the original tomcosmos convention since M0.

The runner converts at the I/O boundary (km, km/s, seconds-since-epoch).
This module lets it ask the sim "what's your time unit in seconds?" and
"what factor turns AU/<your-time-unit> into km/s?" without hard-coding
either choice. Adding a new mode (or changing units) only requires
extending this lookup; the runner stays mode-agnostic.

The conversion factors lean on astropy's unit machinery for the actual
arithmetic — we just translate REBOUND's string names into astropy
units and divide.
"""
from __future__ import annotations

import rebound
from astropy import units as u
from astropy.time import Time

# J2000 reference epoch — TDB midday on 2000-01-01. ASSIST's kernels
# index time as days past this instant.
_J2000_TDB = Time("2000-01-01T12:00:00", scale="tdb")

# Translation from REBOUND unit-name strings to astropy units. REBOUND
# normalizes whatever you pass to `sim.units = (...)` into lowercase
# internally, so we keep the keys lowercase here and lowercase the
# lookup. Add more if a new mode introduces them.
_TIME_UNIT_MAP: dict[str, u.Unit] = {
    "yr":  u.yr,
    "day": u.day,
    "s":   u.s,
}
_LENGTH_UNIT_MAP: dict[str, u.Unit] = {
    "au": u.AU,
    "km": u.km,
    "m":  u.m,
}


def _astropy_time_unit(sim: rebound.Simulation) -> u.Unit:
    name = str(sim.units["time"]).lower()
    try:
        return _TIME_UNIT_MAP[name]
    except KeyError as e:
        raise ValueError(
            f"sim.units['time'] is {name!r}; not in tomcosmos's unit map. "
            f"Known: {sorted(_TIME_UNIT_MAP)}"
        ) from e


def _astropy_length_unit(sim: rebound.Simulation) -> u.Unit:
    name = str(sim.units["length"]).lower()
    try:
        return _LENGTH_UNIT_MAP[name]
    except KeyError as e:
        raise ValueError(
            f"sim.units['length'] is {name!r}; not in tomcosmos's unit map. "
            f"Known: {sorted(_LENGTH_UNIT_MAP)}"
        ) from e


def time_unit_in_seconds(sim: rebound.Simulation) -> float:
    """How many SI seconds make up one of the sim's time units.

    Used to convert sample times (kept in seconds at the I/O boundary)
    into the values `sim.integrate()` expects, and to convert the sim's
    `sim.t` back out for telemetry.
    """
    return float((1.0 * _astropy_time_unit(sim)).to(u.s).value)


def velocity_unit_to_kms(sim: rebound.Simulation) -> float:
    """Factor that turns sim's native velocity (length/time) into km/s.

    Used on the StateHistory output side: `p.vx * velocity_unit_to_kms(sim)`
    yields km/s regardless of whether sim is in AU/yr or AU/day.
    """
    length_kms = float((1.0 * _astropy_length_unit(sim)).to(u.km).value)
    return length_kms / time_unit_in_seconds(sim)


def length_unit_to_km(sim: rebound.Simulation) -> float:
    """Factor that turns sim's native length unit into km."""
    return float((1.0 * _astropy_length_unit(sim)).to(u.km).value)


def days_past_j2000(epoch: Time) -> float:
    """TDB days from J2000 — the time index ASSIST uses to look up
    DE440 perturber positions in its force loop. Mode A needs the
    `rebound.Simulation` anchored here at t=0 so subsequent
    `sim.integrate(t_days)` calls map onto real ephemeris epochs.
    """
    return float((epoch.tdb - _J2000_TDB).to(u.day).value)
