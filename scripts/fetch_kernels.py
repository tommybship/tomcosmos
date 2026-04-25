"""Download SPICE / NAIF kernels into `tomcosmos.config.kernel_dir()`.

Usage:
    python scripts/fetch_kernels.py            # base only (DE440s, 32 MB)
    python scripts/fetch_kernels.py jupiter    # base + jup365.bsp (~1.1 GB)
    python scripts/fetch_kernels.py all-moons  # base + every satellite group

Same logic backs the CLI command `tomcosmos fetch-kernels`. Each
download is verified by length (HEAD Content-Length match), and a
`manifest.json` next to the kernels records the SHA256 + URL for
reproducibility (read by the ephemeris loader to detect silent kernel
swaps).
"""
from __future__ import annotations

import sys

from tomcosmos.kernel_fetch import fetch_groups
from tomcosmos.kernels import (
    ALL_GROUPS,
    BASE_GROUP,
    SATELLITE_GROUPS,
    KernelGroup,
    group_by_name,
)


def _parse_includes(args: list[str]) -> list[KernelGroup]:
    if not args:
        return [BASE_GROUP]
    if "all-moons" in args:
        return list(ALL_GROUPS)
    out: list[KernelGroup] = [BASE_GROUP]
    for name in args:
        try:
            out.append(group_by_name(name))
        except KeyError as e:
            print(f"error: {e}", file=sys.stderr)
            print(f"available groups: base, all-moons, {[g.name for g in SATELLITE_GROUPS]}",
                  file=sys.stderr)
            sys.exit(2)
    # Deduplicate while preserving order.
    seen = set()
    deduped = []
    for g in out:
        if g.name in seen:
            continue
        seen.add(g.name)
        deduped.append(g)
    return deduped


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    groups = _parse_includes(args)
    fetch_groups(groups)
    return 0


if __name__ == "__main__":
    sys.exit(main())
