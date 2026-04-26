"""Kernel registry — what ephemeris files we know how to fetch.

A `KernelGroup` bundles one NAIF .bsp file together with the canonical
body names it provides. Users opt in per group: `fetch-kernels` with
no args downloads the base group (DE440s, 32 MB); `--include jupiter`
adds the 1.1 GB Jovian-system kernel; `--include all-moons` grabs
every satellite group.

URLs and sizes are stable parts of NAIF's catalog, but the kernels
themselves are revised occasionally. SHA pinning lives in
`data/kernels/manifest.json` (committed) and is verified at load
time by `EphemerisSource` — see PLAN.md > "Kernel locking."
"""
from __future__ import annotations

from dataclasses import dataclass, field

NAIF_BASE = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk"
JPL_SSD_ASTEROIDS_BASE = "https://ssd.jpl.nasa.gov/ftp/eph/small_bodies/asteroids_de441"


@dataclass(frozen=True)
class KernelGroup:
    """One downloadable kernel file plus the bodies it provides."""

    name: str                              # short id used for --include
    filename: str                          # .bsp filename on disk + at NAIF
    url: str                               # full download URL
    approx_size_mb: float                  # rounded; for "this is N MB" warnings
    bodies: tuple[str, ...] = field(default_factory=tuple)  # canonical names


# Fetched by default — small and required for the baseline sun-planets scenario.
BASE_GROUP = KernelGroup(
    name="base",
    filename="de440s.bsp",
    url=f"{NAIF_BASE}/planets/de440s.bsp",
    approx_size_mb=32.0,
    bodies=(
        "sun", "mercury", "venus", "earth", "moon",
        "mars", "jupiter", "saturn", "uranus", "neptune",
    ),
)

# Optional satellite groups — opt-in via fetch-kernels --include <name>.
# The set of available NAIF generic-kernel groups is narrower than you'd
# guess: Uranus has no public satellite ephemeris in the generic-kernels
# collection (Voyager-era data lives in PDS, not redistributable here),
# so Titania/Oberon are reachable only via explicit r/v ICs in scenarios
# until a kernel source surfaces. Pluto and Mars-moons are in the catalog
# but we haven't added their bodies to BODY_CONSTANTS yet — easy to add
# alongside a `pluto` / `mars-moons` group when first needed.
SATELLITE_GROUPS: tuple[KernelGroup, ...] = (
    KernelGroup(
        name="mars",
        filename="mar099.bsp",
        url=f"{NAIF_BASE}/satellites/mar099.bsp",
        approx_size_mb=1228.0,  # Phobos's 7-hr orbit needs dense Chebyshev coeffs
        bodies=("phobos", "deimos"),
    ),
    KernelGroup(
        name="jupiter",
        filename="jup365.bsp",
        url=f"{NAIF_BASE}/satellites/jup365.bsp",
        approx_size_mb=1137.0,
        bodies=("io", "europa", "ganymede", "callisto"),
    ),
    # NAIF's higher-numbered sat/nep files (sat45x, nep10x) are NOT
    # newer versions of sat441 / nep097 — they're complementary catalogs
    # of newly-discovered irregular satellites that *require* the base
    # major-moon kernel to also be loaded. sat459 ships only seven
    # 2020-2023 irregulars (NAIF 65297-65303); nep105 ships only
    # Nereid (802). Both omit the bodies M2 actually wants.
    KernelGroup(
        name="saturn",
        filename="sat441.bsp",   # major moons: 601-609 + 612-634 + Saturn (699)
        url=f"{NAIF_BASE}/satellites/sat441.bsp",
        approx_size_mb=631.0,
        # Restricted to bodies present in BODY_CONSTANTS today; sat441 also
        # carries Hyperion (607), Phoebe (609), Helene/Telesto/Calypso/
        # Methone/Polydeuces (612-634). Add their constants when first needed.
        bodies=("mimas", "enceladus", "tethys", "dione",
                "rhea", "titan", "iapetus"),
    ),
    KernelGroup(
        name="neptune",
        filename="nep097.bsp",   # Triton (801) + Neptune (899)
        url=f"{NAIF_BASE}/satellites/nep097.bsp",
        approx_size_mb=100.0,
        bodies=("triton",),
    ),
    KernelGroup(
        name="pluto",
        filename="plu060.bsp",
        url=f"{NAIF_BASE}/satellites/plu060.bsp",
        approx_size_mb=135.0,
        bodies=("pluto", "charon", "nix", "hydra", "kerberos", "styx"),
    ),
)

# ASSIST (Mode A) kernels — full DE440 + 16 large-asteroid perturbers. ASSIST
# accepts the small DE440s for short-span tests, but its upstream test suite
# and our `assist` marker tests both assume the full file, so this is what
# `--include assist` fetches. Set `TOMCOSMOS_ASSIST_PLANET_KERNEL` /
# `TOMCOSMOS_ASSIST_ASTEROID_KERNEL` to point at alternates if you keep them
# elsewhere on disk. NAIF also ships `de441.bsp` covering -13200..+17191 CE
# at ~3.3 GB if you need deep-time propagation; the default below is the
# standard `de440.bsp` covering 1849..2150 CE at ~120 MB, which is plenty
# for mission / NEO / asteroid work.
ASSIST_GROUPS: tuple[KernelGroup, ...] = (
    KernelGroup(
        name="assist-planets",
        filename="de440.bsp",
        url=f"{NAIF_BASE}/planets/de440.bsp",
        approx_size_mb=120.0,
        bodies=(),  # Mode A reads this directly in the force loop, not by name
    ),
    KernelGroup(
        name="assist-asteroids",
        filename="sb441-n16.bsp",
        url=f"{JPL_SSD_ASTEROIDS_BASE}/sb441-n16.bsp",
        approx_size_mb=645.0,  # 16 asteroids over DE441's full ~30 ka span
        bodies=(),
    ),
)

ALL_GROUPS: tuple[KernelGroup, ...] = (BASE_GROUP, *SATELLITE_GROUPS, *ASSIST_GROUPS)


def group_for_body(canonical_name: str) -> KernelGroup | None:
    """Return the group whose kernel provides this body, or None if unknown."""
    for g in ALL_GROUPS:
        if canonical_name in g.bodies:
            return g
    return None


def group_by_name(name: str) -> KernelGroup:
    for g in ALL_GROUPS:
        if g.name == name:
            return g
    raise KeyError(
        f"unknown kernel group {name!r}; "
        f"known: {sorted(g.name for g in ALL_GROUPS)}"
    )
