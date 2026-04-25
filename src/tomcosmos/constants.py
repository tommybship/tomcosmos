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
    # --- Sun + 8 planets + Earth's Moon (M1) ---
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
    # --- M2: Major moons (NASA fact sheets / JPL Solar System Dynamics) ---
    # Galilean moons of Jupiter; SPICE IDs 5xx
    BodyConstant("io",       8.9319e22, 1821.6,   "#E6C68F", 501),
    BodyConstant("europa",   4.7998e22, 1560.8,   "#C2A88B", 502),
    BodyConstant("ganymede", 1.4819e23, 2634.1,   "#9E8A6E", 503),
    BodyConstant("callisto", 1.0759e23, 2410.3,   "#5B5048", 504),
    # Major Saturnian moons; SPICE IDs 6xx
    BodyConstant("mimas",    3.7493e19, 198.2,    "#BDBDBD", 601),
    BodyConstant("enceladus", 1.0802e20, 252.1,   "#E8F4FF", 602),
    BodyConstant("tethys",   6.1745e20, 533.0,    "#D8D8D8", 603),
    BodyConstant("dione",    1.0955e21, 561.7,    "#C9C9C9", 604),
    BodyConstant("rhea",     2.3065e21, 763.8,    "#BFBFBF", 605),
    BodyConstant("titan",    1.3452e23, 2574.7,   "#D8AB6E", 606),
    BodyConstant("iapetus",  1.8056e21, 734.5,    "#A28675", 608),
    # Largest Uranian moon; SPICE IDs 7xx (no NAIF-public satellite kernel —
    # use explicit r/v ICs in scenarios until one becomes available).
    BodyConstant("titania",  3.4000e21, 788.9,    "#A8A89A", 703),
    BodyConstant("oberon",   3.0760e21, 761.4,    "#928579", 704),
    # Largest Neptunian moon; SPICE ID 8xx
    BodyConstant("triton",   2.1390e22, 1353.4,   "#D4C8B3", 801),
    # Mars's moons; SPICE IDs 4xx
    BodyConstant("phobos",   1.0659e16, 11.2667,  "#9C8770", 401),
    BodyConstant("deimos",   1.4762e15, 6.2,      "#A99A82", 402),
    # Pluto + Charon system; SPICE IDs 9xx
    BodyConstant("pluto",    1.303e22,  1188.3,   "#B89F7B", 999, aliases=("pluto-center",)),
    BodyConstant("charon",   1.586e21,  606.0,    "#928879", 901),
    BodyConstant("nix",      4.5e16,    49.8,     "#BDB6A8", 902),
    BodyConstant("hydra",    4.8e16,    50.9,     "#BDB6A8", 903),
    BodyConstant("kerberos", 1.65e16,   19.0,     "#A89F90", 904),
    BodyConstant("styx",     7.5e15,    16.0,     "#A89F90", 905),
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
