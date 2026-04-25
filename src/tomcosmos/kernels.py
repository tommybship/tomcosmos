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
SATELLITE_GROUPS: tuple[KernelGroup, ...] = (
    KernelGroup(
        name="jupiter",
        filename="jup365.bsp",
        url=f"{NAIF_BASE}/satellites/jup365.bsp",
        approx_size_mb=1137.0,
        bodies=("io", "europa", "ganymede", "callisto"),
    ),
    KernelGroup(
        name="saturn",
        filename="sat441.bsp",
        url=f"{NAIF_BASE}/satellites/sat441.bsp",
        approx_size_mb=662.0,
        bodies=("mimas", "enceladus", "tethys", "dione",
                "rhea", "titan", "iapetus"),
    ),
    KernelGroup(
        name="uranus",
        filename="ura111.bsp",
        url=f"{NAIF_BASE}/satellites/ura111.bsp",
        approx_size_mb=2.0,  # placeholder; refined when we actually fetch
        bodies=("titania", "oberon"),
    ),
    KernelGroup(
        name="neptune",
        filename="nep097.bsp",
        url=f"{NAIF_BASE}/satellites/nep097.bsp",
        approx_size_mb=2.0,  # placeholder
        bodies=("triton",),
    ),
)

ALL_GROUPS: tuple[KernelGroup, ...] = (BASE_GROUP, *SATELLITE_GROUPS)


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
