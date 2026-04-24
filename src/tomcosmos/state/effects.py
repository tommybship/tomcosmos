"""Optional physics effects — additional forces bolted on top of pure N-body.

Each function attaches its effect to a `rebound.Simulation` and returns
something the caller can keep alive for the sim's lifetime. REBOUND's
C side only holds weak references to Python-level callbacks and to
REBOUNDx's Extras handle; if Python GCs the underlying object, the
next force evaluation segfaults. `build_simulation` stashes the
returned handle on the sim as a hidden attribute.

Currently implemented:
  - `gr` — 1PN Einstein correction (Schwarzschild) from the Sun as the
    dominant mass. Recovers Mercury's perihelion precession
    (~43 arcsec/century).

Backend selection for `gr`:
  - If `reboundx` imports successfully → use REBOUNDx's `gr` force
    (battle-tested, same C code the rest of the REBOUND community
    uses).
  - Otherwise → use our own Python implementation via REBOUND's
    `additional_forces` hook. Cross-platform, no dependency,
    verified to agree with REBOUNDx to within integrator roundoff.

REBOUNDx doesn't build on Windows with MSVC out of the box. We maintain
a patched fork at https://github.com/tommybship/reboundx (branch
`windows-msvc-build`) until the fix merges upstream. Linux/macOS users
get REBOUNDx from PyPI normally.

Formula (Einstein-Infeld-Hoffmann simplified for a dominant central mass
M_sun; see Will 1993, "Theory and Experiment in Gravitational Physics",
§6.2, or Newhall/Standish/Williams DE-series ephemeris write-ups):

    a_GR = (GM_sun / (c² r³)) *
           [ (4 * GM_sun / r - v²) * r_vec + 4 * (r·v) * v_vec ]

where r_vec and v_vec are the particle's position and velocity relative
to the Sun, r = |r_vec|, v² = |v_vec|². The force is velocity-dependent,
so the sim must be flagged `force_is_velocity_dependent = 1` for WHFast
to handle it correctly. REBOUNDx sets that flag itself.
"""
from __future__ import annotations

from typing import Literal

import rebound

try:
    import reboundx as _reboundx
    HAS_REBOUNDX: bool = True
except ImportError:
    _reboundx = None
    HAS_REBOUNDX = False

EffectName = Literal["gr"]

# Speed of light in AU / yr. Derived from IAU 2012 AU (149,597,870,700 m),
# Julian year (31,557,600 s), c = 299,792,458 m/s.
_C_AU_PER_YR: float = 63239.7263263


def attach_gr(sim: rebound.Simulation, sun_hash: int) -> object:
    """Register a 1PN Einstein correction with `sim`.

    Prefers REBOUNDx's `gr` force when it's importable; otherwise falls
    back to the Python implementation below. Both treat the Sun as the
    dominant mass — REBOUNDx uses particle index 0, our Python version
    uses the particle whose hash matches `sun_hash`.

    Returns a handle the caller must keep alive (Extras in the
    REBOUNDx case, a bound function in the fallback case). Don't let
    it be garbage-collected; the next force evaluation crashes.
    """
    if HAS_REBOUNDX:
        return _attach_gr_reboundx(sim)
    return _attach_gr_python(sim, sun_hash)


def _attach_gr_reboundx(sim: rebound.Simulation) -> object:
    """REBOUNDx backend — equivalent of our Python implementation but
    in compiled C. Treats particle 0 as the source mass; M1 scenarios
    put the Sun first, so this is consistent with our convention."""
    rebx = _reboundx.Extras(sim)
    gr = rebx.load_force("gr")
    gr.params["c"] = _C_AU_PER_YR
    rebx.add_force(gr)
    return rebx  # keep Extras alive for sim lifetime


def _attach_gr_python(sim: rebound.Simulation, sun_hash: int) -> object:
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
