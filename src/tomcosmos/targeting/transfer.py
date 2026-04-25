"""Compute the two Δvs that move a spacecraft from one body to another.

The composition pattern: query the ephemeris source for the bodies'
heliocentric states at the departure and arrival epochs, hand
`(r_dep, r_arr, tof)` to the Lambert solver, get the two velocities
that define the transfer arc, subtract from each body's instantaneous
velocity to get the impulsive Δvs.

The result is a `Transfer` dataclass that packages the math; users
turn it into scenario `DeltaV` events with `.as_dv_events(scenario_epoch)`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from astropy import units as u
from astropy.time import Time

from tomcosmos.state.ephemeris import EphemerisSource
from tomcosmos.state.scenario import DeltaV
from tomcosmos.targeting.lambert import lambert

# Heliocentric central-body parameter — Sun's GM in km³/s². Matches the
# value used in tests/test_targeting_lambert.py and the rest of the
# codebase's km/s/kg convention.
MU_SUN_KM3_S2 = 1.32712440018e11


@dataclass(frozen=True)
class Transfer:
    """The two-impulse transfer between `from_body` at `departure_epoch`
    and `to_body` at `arrival_epoch`.

    All vectors are in ICRF barycentric, km/s. The Δvs are what a
    spacecraft *riding the departure body* must add to match the
    transfer-arc velocity, and what it must add at arrival to match
    the destination body's velocity. Both can be plugged into a
    scenario's `dv_events` list directly.
    """

    from_body: str
    to_body: str
    departure_epoch: Time
    arrival_epoch: Time
    tof_s: float
    v_at_departure: np.ndarray  # heliocentric velocity on the transfer arc at departure
    v_at_arrival: np.ndarray    # heliocentric velocity on the transfer arc at arrival
    delta_v_departure: np.ndarray  # what the spacecraft adds at departure
    delta_v_arrival: np.ndarray    # what the spacecraft adds at arrival

    def as_dv_events(self, scenario_epoch: Time) -> tuple[DeltaV, DeltaV]:
        """Return the two `DeltaV` events ready to attach to a scenario.

        `scenario_epoch` is the run's t=0; `t_offset` for each burn is
        measured from there. The schema rejects t_offset = 0 (the
        Duration parser requires positive values), so the departure
        burn is offset by 1 second — small enough to be physically
        equivalent to "at scenario epoch" but valid against the
        validator. Adjust if your scenario's epoch differs from the
        intended departure epoch by more than a few seconds.
        """
        dep_offset_s = max(1.0, float((self.departure_epoch - scenario_epoch).to(u.s).value))
        arr_offset_s = float((self.arrival_epoch - scenario_epoch).to(u.s).value)
        return (
            DeltaV.model_validate({
                "t_offset": f"{dep_offset_s} s",
                "dv": tuple(float(c) for c in self.delta_v_departure),
                "frame": "icrf_barycentric",
            }),
            DeltaV.model_validate({
                "t_offset": f"{arr_offset_s} s",
                "dv": tuple(float(c) for c in self.delta_v_arrival),
                "frame": "icrf_barycentric",
            }),
        )


def compute_transfer(
    source: EphemerisSource,
    from_body: str,
    to_body: str,
    departure_epoch: Time,
    arrival_epoch: Time,
    *,
    mu: float = MU_SUN_KM3_S2,
    prograde: bool = True,
) -> Transfer:
    """Build a `Transfer` connecting `from_body` at `departure_epoch`
    to `to_body` at `arrival_epoch`.

    Both bodies are looked up via the ephemeris source. The transfer
    is solved as a single-revolution Lambert in the heliocentric
    reference (μ = MU_SUN by default; override for moon-system or
    lunar-transfer use). `prograde=True` chooses the short-way arc
    (counterclockwise around +z) — the natural choice for transfers
    in the ecliptic plane between prograde-orbiting bodies.

    Returns a `Transfer` dataclass holding the burns and arc states;
    use `.as_dv_events(scenario.epoch)` to obtain ready-to-attach
    scenario events.
    """
    if arrival_epoch <= departure_epoch:
        raise ValueError(
            f"arrival_epoch ({arrival_epoch.isot}) must be strictly after "
            f"departure_epoch ({departure_epoch.isot})"
        )

    r_from, v_from = source.query(from_body, departure_epoch)
    r_to, v_to = source.query(to_body, arrival_epoch)

    tof_s = float((arrival_epoch - departure_epoch).to(u.s).value)
    v1, v2 = lambert(r_from, r_to, tof_s, mu, prograde=prograde)

    return Transfer(
        from_body=from_body,
        to_body=to_body,
        departure_epoch=departure_epoch,
        arrival_epoch=arrival_epoch,
        tof_s=tof_s,
        v_at_departure=v1,
        v_at_arrival=v2,
        delta_v_departure=v1 - v_from,
        delta_v_arrival=v_to - v2,
    )
