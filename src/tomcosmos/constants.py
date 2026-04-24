"""Physical constants for bodies tomcosmos knows about.

Values are from NASA planetary fact sheets (Williams, 2024 revision).
Four significant figures is enough: we pay a ~1e-4 relative ceiling to
using mass * G rather than JPL GM anyway (see PLAN.md > Non-goals), so
polishing these to ten digits wouldn't buy real accuracy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from difflib import get_close_matches

from tomcosmos.exceptions import UnknownBodyError


@dataclass(frozen=True)
class BodyConstant:
    name: str
    mass_kg: float
    radius_km: float
    color_hex: str
    spice_id: int
    aliases: tuple[str, ...] = field(default_factory=tuple)


_RAW: tuple[BodyConstant, ...] = (
    BodyConstant("sun",     1.989e30,   695700.0, "#F5C518", 10,  aliases=("sol",)),
    BodyConstant("mercury", 3.3011e23,  2439.7,   "#8C7853", 199),
    BodyConstant("venus",   4.8675e24,  6051.8,   "#E8B273", 299),
    BodyConstant("earth",   5.9724e24,  6371.0,   "#4A90D9", 399, aliases=("terra",)),
    BodyConstant("moon",    7.3420e22,  1737.4,   "#C0C0C0", 301, aliases=("luna",)),
    BodyConstant("mars",    6.4171e23,  3389.5,   "#CD5C5C", 499),
    BodyConstant("jupiter", 1.8982e27,  69911.0,  "#D2A567", 599),
    BodyConstant("saturn",  5.6834e26,  58232.0,  "#F4E4A1", 699),
    BodyConstant("uranus",  8.6810e25,  25362.0,  "#AFEEEE", 799),
    BodyConstant("neptune", 1.02413e26, 24622.0,  "#4169E1", 899),
)

BODY_CONSTANTS: dict[str, BodyConstant] = {b.name: b for b in _RAW}
_BY_SPICE_ID: dict[int, BodyConstant] = {b.spice_id: b for b in _RAW}
_BY_ALIAS: dict[str, BodyConstant] = {
    alias.lower(): b for b in _RAW for alias in b.aliases
}


def resolve_body_constant(key: str | int) -> BodyConstant:
    """Look up a body by name (case-insensitive), alias, or SPICE ID.

    Raises UnknownBodyError with a 'did you mean' suggestion on miss.
    """
    if isinstance(key, bool):
        raise TypeError(f"key must be str or int, got bool ({key!r})")
    if isinstance(key, int):
        if key in _BY_SPICE_ID:
            return _BY_SPICE_ID[key]
        raise UnknownBodyError(
            f"unknown SPICE ID {key}; known: {sorted(_BY_SPICE_ID)}"
        )
    if isinstance(key, str):
        k = key.lower().strip()
        if k in BODY_CONSTANTS:
            return BODY_CONSTANTS[k]
        if k in _BY_ALIAS:
            return _BY_ALIAS[k]
        suggestions = get_close_matches(
            k, list(BODY_CONSTANTS) + list(_BY_ALIAS), n=3, cutoff=0.5
        )
        msg = f"unknown body {key!r}"
        if suggestions:
            msg += f"; did you mean: {', '.join(suggestions)}?"
        raise UnknownBodyError(msg)
    raise TypeError(f"key must be str or int, got {type(key).__name__}")
