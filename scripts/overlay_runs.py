"""Open the viewer with two runs overlaid.

Reads two Parquet files, renames the second run's bodies with a suffix
(default ``*``), concatenates them into one StateHistory, and opens the
existing pyvista Viewer. Bodies whose names match BODY_CONSTANTS render
with their canonical colors; suffixed names fall back to grey, which
visually marks the overlay.

Both runs must share the same epoch and cadence so sample_idx aligns —
the script asserts this.

Usage:
    python scripts/overlay_runs.py runs/sun-planets-100yr__*.parquet \\
                                   runs/jupiter-2x-mass__*.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from tomcosmos.io.history import StateHistory
from tomcosmos.viz.pyvista_viewer import Viewer


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)

    base_path = Path(sys.argv[1])
    overlay_path = Path(sys.argv[2])
    suffix = sys.argv[3] if len(sys.argv) > 3 else "*"

    base = StateHistory.from_parquet(base_path)
    overlay = StateHistory.from_parquet(overlay_path)

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
    overlay_df["body"] = overlay_df["body"].astype(str) + suffix

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
        f"{len(overlay.body_names)} bodies (grey, suffix={suffix!r})"
    )
    Viewer(merged).show()


if __name__ == "__main__":
    main()
