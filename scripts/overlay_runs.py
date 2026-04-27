"""Open the viewer with two runs overlaid.

Reads two Parquet files, renames the second run's bodies with a suffix
(default ``*``), concatenates them into one StateHistory, and opens the
existing pyvista Viewer. Bodies whose names match BODY_CONSTANTS render
with their canonical colors; suffixed names fall back to grey, which
visually marks the overlay.

Both runs must share the same epoch and cadence so sample_idx aligns —
the script asserts this.

Usage:
    python scripts/overlay_runs.py [--follow BODY] [--suffix S] BASE OVERLAY

Example:
    python scripts/overlay_runs.py --follow earth \\
        runs/apophis-2029-flyby-context__*.parquet \\
        runs/apophis-2029-flyby__*.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from tomcosmos.io.history import StateHistory
from tomcosmos.viz.pyvista_viewer import Viewer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("base", type=Path, help="canonical-color run")
    parser.add_argument("overlay", type=Path, help="grey-overlay run")
    parser.add_argument(
        "--follow", default=None,
        help="Body name to keep centered in the viewport (e.g. earth).",
    )
    parser.add_argument(
        "--scaling", default="log", choices=("log", "true", "marker"),
        help="Body size scaling. 'log' (default) exaggerates radii so all "
             "bodies are visible at solar-system zoom. 'true' renders at "
             "physical size — necessary for close-up flybys where log "
             "scaling makes bodies engulf each other.",
    )
    parser.add_argument(
        "--suffix", default="*",
        help="Suffix appended to overlay body names so they render as grey "
             "(unknown to BODY_CONSTANTS). Default '*'.",
    )
    args = parser.parse_args()

    base = StateHistory.from_parquet(args.base)
    overlay = StateHistory.from_parquet(args.overlay)

    if base.scenario.epoch.tdb.jd != overlay.scenario.epoch.tdb.jd:
        raise SystemExit(
            f"epoch mismatch: {base.scenario.epoch} vs {overlay.scenario.epoch}"
        )
    if base.df["sample_idx"].max() != overlay.df["sample_idx"].max():
        raise SystemExit(
            f"sample count mismatch: {base.df['sample_idx'].max() + 1} "
            f"vs {overlay.df['sample_idx'].max() + 1}"
        )

    overlay_df = overlay.df.copy()
    overlay_df["body"] = overlay_df["body"].astype(str) + args.suffix

    merged_df = pd.concat([base.df, overlay_df], ignore_index=True)
    merged_names = tuple(merged_df["body"].astype(str).unique().tolist())

    merged = StateHistory(
        df=merged_df,
        scenario=base.scenario,
        body_names=merged_names,
        metadata=None,
    )

    print(
        f"overlay: {len(base.body_names)} bodies (colored) + "
        f"{len(overlay.body_names)} bodies (grey, suffix={args.suffix!r})"
    )
    Viewer(merged, follow=args.follow, scaling=args.scaling).show()


if __name__ == "__main__":
    main()
