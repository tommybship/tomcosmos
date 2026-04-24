"""Optional physics effects — additional forces bolted on top of pure N-body.

Each function attaches a callback to a `rebound.Simulation` and returns
the Python callable that REBOUND will invoke during force evaluations.
The returned callable must be kept alive for as long as the simulation
runs (REBOUND stores a C wrapper around it but the underlying Python
object also needs a live reference). `build_simulation` stashes them
on the sim as a hidden attribute.

Currently implemented:
  - `gr` — 1PN Einstein correction (Schwarzschild) from the Sun as the
    dominant mass. Recovers Mercury's perihelion precession
    (~43 arcsec/century). Cross-platform (no REBOUNDx dependency).

Formula (Einstein-Infeld-Hoffmann simplified for a dominant central mass
M_sun; see Will 1993, "Theory and Experiment in Gravitational Physics",
§6.2, or Newhall/Standish/Williams DE-series ephemeris write-ups):

    a_GR = (GM_sun / (c² r³)) *
           [ (4 * GM_sun / r - v²) * r_vec + 4 * (r·v) * v_vec ]

where r_vec and v_vec are the particle's position and velocity relative
to the Sun, r = |r_vec|, v² = |v_vec|². This is a velocity-dependent
force, so the sim must be flagged with `force_is_velocity_dependent = 1`
for WHFast to handle it correctly.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import rebound

EffectName = Literal["gr"]

# Speed of light in AU / yr. Derived from IAU 2012 AU (149,597,870,700 m)
# and Julian year (31,557,600 s) with c = 299,792,458 m/s.
_C_AU_PER_YR: float = 63239.7263263


def attach_gr(sim: rebound.Simulation, sun_hash: int) -> Callable[[object], None]:
    """Register a 1PN Einstein correction with `sim`, treating the particle
    identified by `sun_hash` as the dominant mass.

    Returns the Python callback so the caller can keep a reference alive —
    REBOUND's C-side only holds a CFUNCTYPE wrapper; if Python GCs the
    underlying function, the next force evaluation segfaults.
    """
    c2 = _C_AU_PER_YR * _C_AU_PER_YR
    sun_hash_u32 = int(sun_hash) & 0xFFFFFFFF

    def _gr_force(sim_ptr: object) -> None:
        s = sim_ptr.contents  # type: ignore[attr-defined]
        ps = s.particles

        # Find Sun by hash. Small N (< ~50 for M1-M4); linear scan is fine.
        sun_idx = -1
        for i in range(s.N):
            if ps[i].hash.value == sun_hash_u32:
                sun_idx = i
                break
        if sun_idx < 0:
            return  # Sun removed (shouldn't happen) — skip silently
        sun = ps[sun_idx]
        gm_sun = s.G * sun.m

        sx, sy, sz = sun.x, sun.y, sun.z
        svx, svy, svz = sun.vx, sun.vy, sun.vz

        for i in range(s.N):
            if i == sun_idx:
                continue
            p = ps[i]
            dx = p.x - sx
            dy = p.y - sy
            dz = p.z - sz
            dvx = p.vx - svx
            dvy = p.vy - svy
            dvz = p.vz - svz

            r2 = dx * dx + dy * dy + dz * dz
            r = r2 ** 0.5
            v2 = dvx * dvx + dvy * dvy + dvz * dvz
            rdotv = dx * dvx + dy * dvy + dz * dvz

            # a_GR = (GM/(c² r³)) * [ (4GM/r - v²) r + 4(r·v) v ]
            prefac = gm_sun / (c2 * r2 * r)
            radial = 4.0 * gm_sun / r - v2
            ax = prefac * (radial * dx + 4.0 * rdotv * dvx)
            ay = prefac * (radial * dy + 4.0 * rdotv * dvy)
            az = prefac * (radial * dz + 4.0 * rdotv * dvz)
            p.ax += ax
            p.ay += ay
            p.az += az

    sim.additional_forces = _gr_force
    sim.force_is_velocity_dependent = 1
    return _gr_force
