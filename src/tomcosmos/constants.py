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
    # IAU 2015 rotational elements (Archinal et al. 2015 "Report of the
    # IAU Working Group on Cartographic Coordinates and Rotational
    # Elements: 2015"). Used by the viewer to spin textured bodies on
    # their physical axes:
    #   ICRF → body-fixed = R_z(W) · R_x(90°−δ₀) · R_z(α₀+90°)
    #   W(t) = prime_meridian_at_j2000_deg + rotation_rate_deg_per_day × Δd
    # where Δd = days since J2000.0 TDB.
    #
    # Negative `rotation_rate_deg_per_day` is retrograde (Venus, Uranus,
    # Pluto). Bodies without published IAU rotation (test particles,
    # smaller asteroids) leave all four fields None and don't spin.
    #
    # Time-dependent terms (slow precession of α₀/δ₀, periodic
    # librations) are not modeled — they're sub-degree over century
    # scales and invisible at viewer zoom. The viewer uses these J2000
    # values directly.
    pole_ra_deg: float | None = None
    pole_dec_deg: float | None = None
    prime_meridian_at_j2000_deg: float | None = None
    rotation_rate_deg_per_day: float | None = None


_RAW: tuple[BodyConstant, ...] = (
    # --- Sun + 8 planets + Earth's Moon (M1) ---
    # IAU rotational elements (Archinal et al. 2015) included for every
    # textured body — viewer uses them to spin meshes on physical axes.
    BodyConstant(
        "sun",     1.989e30,   695700.0, "#F5C518", 10,  aliases=("sol",),
        pole_ra_deg=286.13, pole_dec_deg=63.87,
        prime_meridian_at_j2000_deg=84.176, rotation_rate_deg_per_day=14.1844,
    ),
    BodyConstant(
        "mercury", 3.3011e23,  2439.7,   "#8C7853", 199,
        pole_ra_deg=281.0103, pole_dec_deg=61.4155,
        prime_meridian_at_j2000_deg=329.5988, rotation_rate_deg_per_day=6.1385108,
    ),
    BodyConstant(
        "venus",   4.8675e24,  6051.8,   "#E8B273", 299,
        pole_ra_deg=272.76, pole_dec_deg=67.16,
        prime_meridian_at_j2000_deg=160.20, rotation_rate_deg_per_day=-1.4813688,
    ),
    BodyConstant(
        "earth",   5.9724e24,  6371.0,   "#4A90D9", 399, aliases=("terra",),
        pole_ra_deg=0.00, pole_dec_deg=90.00,
        prime_meridian_at_j2000_deg=190.147, rotation_rate_deg_per_day=360.9856235,
    ),
    BodyConstant(
        "moon",    7.3420e22,  1737.4,   "#C0C0C0", 301, aliases=("luna",),
        pole_ra_deg=269.9949, pole_dec_deg=66.5392,
        prime_meridian_at_j2000_deg=38.3213, rotation_rate_deg_per_day=13.17635815,
    ),
    BodyConstant(
        "mars",    6.4171e23,  3389.5,   "#CD5C5C", 499,
        pole_ra_deg=317.68143, pole_dec_deg=52.88650,
        prime_meridian_at_j2000_deg=176.630, rotation_rate_deg_per_day=350.89198226,
    ),
    BodyConstant(
        "jupiter", 1.8982e27,  69911.0,  "#D2A567", 599,
        pole_ra_deg=268.056595, pole_dec_deg=64.495303,
        prime_meridian_at_j2000_deg=284.95, rotation_rate_deg_per_day=870.5360000,
    ),
    BodyConstant(
        "saturn",  5.6834e26,  58232.0,  "#F4E4A1", 699,
        pole_ra_deg=40.589, pole_dec_deg=83.537,
        prime_meridian_at_j2000_deg=38.90, rotation_rate_deg_per_day=810.7939024,
    ),
    BodyConstant(
        "uranus",  8.6810e25,  25362.0,  "#AFEEEE", 799,
        pole_ra_deg=257.311, pole_dec_deg=-15.175,
        prime_meridian_at_j2000_deg=203.81, rotation_rate_deg_per_day=-501.1600928,
    ),
    BodyConstant(
        "neptune", 1.02413e26, 24622.0,  "#4169E1", 899,
        pole_ra_deg=299.36, pole_dec_deg=43.46,
        prime_meridian_at_j2000_deg=253.18, rotation_rate_deg_per_day=536.3128492,
    ),
    # --- M2: Major moons (NASA fact sheets / JPL Solar System Dynamics) ---
    # Galilean moons of Jupiter; SPICE IDs 5xx. IAU 2015 rotational
    # elements; all four are tidally locked to Jupiter (spin period =
    # orbital period).
    BodyConstant(
        "io",       8.9319e22, 1821.6,   "#E6C68F", 501,
        pole_ra_deg=268.05, pole_dec_deg=64.50,
        prime_meridian_at_j2000_deg=200.39, rotation_rate_deg_per_day=203.4889538,
    ),
    BodyConstant(
        "europa",   4.7998e22, 1560.8,   "#C2A88B", 502,
        pole_ra_deg=268.08, pole_dec_deg=64.51,
        prime_meridian_at_j2000_deg=36.022, rotation_rate_deg_per_day=101.3747235,
    ),
    BodyConstant(
        "ganymede", 1.4819e23, 2634.1,   "#9E8A6E", 503,
        pole_ra_deg=268.20, pole_dec_deg=64.57,
        prime_meridian_at_j2000_deg=44.064, rotation_rate_deg_per_day=50.3176081,
    ),
    BodyConstant(
        "callisto", 1.0759e23, 2410.3,   "#5B5048", 504,
        pole_ra_deg=268.72, pole_dec_deg=64.83,
        prime_meridian_at_j2000_deg=259.51, rotation_rate_deg_per_day=21.5710715,
    ),
    # Major Saturnian moons; SPICE IDs 6xx
    BodyConstant("mimas",    3.7493e19, 198.2,    "#BDBDBD", 601),
    BodyConstant("enceladus", 1.0802e20, 252.1,   "#E8F4FF", 602),
    BodyConstant("tethys",   6.1745e20, 533.0,    "#D8D8D8", 603),
    BodyConstant("dione",    1.0955e21, 561.7,    "#C9C9C9", 604),
    BodyConstant("rhea",     2.3065e21, 763.8,    "#BFBFBF", 605),
    BodyConstant(
        "titan",    1.3452e23, 2574.7,   "#D8AB6E", 606,
        pole_ra_deg=39.4827, pole_dec_deg=83.4279,
        prime_meridian_at_j2000_deg=186.5855, rotation_rate_deg_per_day=22.5769768,
    ),
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
    BodyConstant(
        "pluto",    1.303e22,  1188.3,   "#B89F7B", 999, aliases=("pluto-center",),
        pole_ra_deg=132.993, pole_dec_deg=-6.163,
        prime_meridian_at_j2000_deg=302.695, rotation_rate_deg_per_day=56.3625225,
    ),
    BodyConstant("charon",   1.586e21,  606.0,    "#928879", 901),
    BodyConstant("nix",      4.5e16,    49.8,     "#BDB6A8", 902),
    BodyConstant("hydra",    4.8e16,    50.9,     "#BDB6A8", 903),
    BodyConstant("kerberos", 1.65e16,   19.0,     "#A89F90", 904),
    BodyConstant("styx",     7.5e15,    16.0,     "#A89F90", 905),
    # --- Selected named asteroids (M5 demos). NAIF IDs follow the
    # SPICE convention 2,000,000 + asteroid number for numbered objects.
    # spice_id is metadata only; asteroid IC comes from explicit r/v in
    # the scenario, not skyfield. Radii / masses from JPL SBDB. ---
    BodyConstant("apophis",  6.1e10,    0.170,    "#7A6F65", 2099942),
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
