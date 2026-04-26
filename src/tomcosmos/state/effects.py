"""Optional physics effects for Mode B (vanilla REBOUND + REBOUNDx).

Mode A (`integrator.ephemeris_perturbers=True`, ASSIST-driven) bakes
GR + J2 into its DE440-driven force model and rejects an `effects`
list at the schema layer; this module is unreachable from that path.

Currently implemented:
  - `gr` — 1PN Einstein correction (Schwarzschild) from the Sun as the
    dominant mass via REBOUNDx's `gr` force. Recovers Mercury's
    perihelion precession (~43 arcsec/century).

REBOUNDx is imported lazily so users running pure Newtonian Mode B —
or any Mode A scenario — don't need it installed. Asking for an
effect when REBOUNDx isn't available raises a clear error pointing
at the install command.

REBOUNDx doesn't build on Windows with MSVC out of the box. We maintain
a patched fork at https://github.com/tommybship/reboundx (branch
`windows-msvc-build`) until the fix merges upstream. Linux/macOS users
get REBOUNDx from PyPI normally.
"""
from __future__ import annotations

from typing import Literal

import rebound

EffectName = Literal["gr"]

# Speed of light in AU / yr. Derived from IAU 2012 AU (149,597,870,700 m),
# Julian year (31,557,600 s), c = 299,792,458 m/s.
_C_AU_PER_YR: float = 63239.7263263


def attach_gr(sim: rebound.Simulation) -> object:
    """Register REBOUNDx's 1PN GR force on `sim`. Sun is particle 0
    (Mode B convention; the schema validator enforces a body named
    "sun" exists when GR is requested).

    Returns the `reboundx.Extras` handle. The caller must keep it alive
    for the simulation's lifetime — REBOUND's C side holds only a
    callback pointer; if Python GCs the Extras object, the next force
    evaluation segfaults.
    """
    try:
        import reboundx
    except ImportError as e:
        raise RuntimeError(
            "integrator.effects=['gr'] requires the `reboundx` package. "
            "Install it with `pip install reboundx` (Linux/macOS) or, on "
            "Windows, our patched fork: "
            "`pip install git+https://github.com/tommybship/reboundx@windows-msvc-build`."
        ) from e

    rebx = reboundx.Extras(sim)
    gr = rebx.load_force("gr")
    gr.params["c"] = _C_AU_PER_YR
    rebx.add_force(gr)
    return rebx
