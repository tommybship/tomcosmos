"""Build a Mode A bulk-asteroid scenario from JPL's NEO catalog.

Queries JPL's SBDB Query API to enumerate numbered NEOs (sorted by
SPK ID, which roughly correlates with discovery order — older
numbered objects come first), then queries Horizons for each
object's ICRF barycentric state at the chosen scenario epoch, and
emits a runnable Mode A scenario YAML with one explicit-IC test
particle per asteroid.

Why Horizons rather than SBDB-Kepler-propagate: SBDB returns each
object's elements at *its own* element epoch (typically months
stale), and Kepler-propagating from there to a common scenario epoch
silently injects perturbation drift Mode A then faithfully reproduces.
For a bulk scenario where every body should agree with the live JPL
prediction, Horizons does the propagation correctly. The disk cache
under `data/cache/horizons_vectors.json` makes subsequent runs free.

Usage (with the tomcosmos env active):

    python scripts/build_neos_scenario.py --count 100 \\
        --epoch "2026-04-26T00:00:00 TDB" \\
        --duration "365 day" \\
        --cadence "5 day" \\
        --out scenarios/neos-100.yaml

The committed `scenarios/neos-100.yaml` was built with these exact
flags. JPL re-fits NEO orbits as new astrometric observations land
— re-running this script overwrites the YAML with the latest
state vectors.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

from astropy.time import Time

from tomcosmos.targeting import horizons

REPO_ROOT = Path(__file__).resolve().parent.parent
SBDB_QUERY_URL = (
    "https://ssd-api.jpl.nasa.gov/sbdb_query.api"
    "?fields=spkid,full_name&sb-group=neo&sb-kind=a&full-prec=1"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--count", type=int, default=100,
        help="How many NEOs to ingest (default 100). The SBDB query "
             "returns objects sorted by SPK ID, so smaller counts pick "
             "the most-historically-observed bodies.",
    )
    parser.add_argument(
        "--epoch", default="2026-04-26T00:00:00 TDB",
        help="Scenario epoch as 'YYYY-MM-DDTHH:MM:SS SCALE' (default 2026-04-26 TDB). "
             "All test-particle states are queried from Horizons at this instant.",
    )
    parser.add_argument(
        "--duration", default="365 day",
        help="Scenario duration (default '365 day').",
    )
    parser.add_argument(
        "--cadence", default="5 day",
        help="Output cadence (default '5 day').",
    )
    parser.add_argument(
        "--out", type=Path,
        default=REPO_ROOT / "scenarios" / "neos-100.yaml",
        help="Output YAML path (default scenarios/neos-100.yaml).",
    )
    args = parser.parse_args()

    epoch = _parse_epoch(args.epoch)

    print(f"Querying SBDB for {args.count} numbered NEOs...")
    designations = _fetch_neo_designations(args.count)
    print(f"  got {len(designations)} designations")

    print(f"Querying Horizons for state vectors at {epoch.isot} TDB...")
    states = horizons.bulk_states_at_epoch(designations, epoch)
    print(f"  got {len(states)} states (cache hits / network mix)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    yaml_text = _render_yaml(
        states=states,
        epoch_str=args.epoch,
        duration=args.duration,
        cadence=args.cadence,
        count=len(states),
    )
    args.out.write_text(yaml_text, encoding="utf-8")
    print(f"Wrote {args.out} ({args.out.stat().st_size / 1024:.1f} KB)")
    return 0


def _parse_epoch(s: str) -> Time:
    """Parse 'YYYY-MM-DDTHH:MM:SS SCALE' the same way scenario YAML does."""
    parts = s.strip().rsplit(" ", 1)
    if len(parts) != 2:
        raise SystemExit(
            f"--epoch must be 'ISO SCALE' (e.g. '2026-04-26T00:00:00 TDB'); got {s!r}"
        )
    iso, scale = parts
    return Time(iso, scale=scale.lower())


def _fetch_neo_designations(count: int) -> list[str]:
    """Hit JPL's SBDB Query API for the first `count` numbered NEOs.

    Returns SPK-ID strings (Horizons accepts these directly via
    `id_type="smallbody"` — no need to round-trip through provisional
    designations).
    """
    url = f"{SBDB_QUERY_URL}&limit={count}"
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            data = json.load(response)
    except urllib.error.HTTPError as e:
        raise SystemExit(
            f"SBDB query failed ({e.code}): {e.read().decode()[:200]}"
        ) from e
    rows = data.get("data") or []
    out: list[str] = []
    for row in rows:
        spkid = str(row[0])
        # SPK IDs in this query come back as 7-digit forms (20000433);
        # Horizons accepts these directly. Strip the leading "200" if
        # present to recover the asteroid number — both work, but the
        # short form keeps designations readable in the YAML.
        if spkid.startswith("2000"):
            asteroid_number = spkid[4:].lstrip("0") or "0"
            out.append(asteroid_number)
        else:
            out.append(spkid)
    return out


def _render_yaml(
    *,
    states: list[horizons.HorizonsState],
    epoch_str: str,
    duration: str,
    cadence: str,
    count: int,
) -> str:
    """Hand-rolled YAML emit. Avoids a yaml.safe_dump round-trip that
    would reorder keys and lose the comment header — the scenario file
    is meant to be readable as data."""
    lines: list[str] = [
        "schema_version: 1",
        f"name: neos-{count}",
        f"# {count}-asteroid Mode A demo scenario.",
        "#",
        "# Test-particle initial conditions were snapshotted from JPL Horizons",
        f"# at the scenario epoch ({epoch_str}). The full N-body propagation",
        "# from each asteroid's orbit-determination element-epoch up to the",
        "# scenario epoch is JPL's, so each particle starts at JPL's authoritative",
        "# best-known state — no Kepler-only IC drift accumulates from stale",
        "# element epochs (see PLAN.md > M5b > Horizons vs SBDB).",
        "#",
        "# Regenerate against the latest Horizons state with:",
        f"#   python scripts/build_neos_scenario.py --count {count} \\",
        f'#       --epoch "{epoch_str}" --duration "{duration}" \\',
        f'#       --cadence "{cadence}"',
        "#",
        "# Requires the ASSIST kernels:",
        "#   tomcosmos fetch-kernels --include assist",
        "",
        f'epoch: "{epoch_str}"',
        f'duration: "{duration}"',
        "integrator:",
        "  name: ias15",
        "  ephemeris_perturbers: true",
        "output:",
        "  format: parquet",
        f'  cadence: "{cadence}"',
        "test_particles:",
    ]
    for s in states:
        # Sanitize the YAML particle name: drop spaces, quote what's left.
        # Horizons returns full_name like "  433 Eros (A898 PA)"; use it
        # so the parquet output preserves human-readable names.
        name = s.target_name.strip()
        lines.append(f"  - name: {json.dumps(name)}")
        lines.append("    ic:")
        lines.append("      type: explicit")
        lines.append("      frame: icrf_barycentric")
        lines.append(
            f"      r: [{float(s.r_km[0])!r}, {float(s.r_km[1])!r}, {float(s.r_km[2])!r}]"
        )
        lines.append(
            f"      v: [{float(s.v_kms[0])!r}, {float(s.v_kms[1])!r}, {float(s.v_kms[2])!r}]"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main())
