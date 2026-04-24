"""StateHistory — in-memory representation of a simulation's output.

Long-format pandas DataFrame: one row per (sample, body). Parquet I/O
lands in a follow-up (part 6); this module defines the shape and a
handful of accessors so everything downstream (analysis, viz) can
point at a stable surface.

Schema (see PLAN.md > "StateHistory schema"):
    sample_idx     int64      0-indexed sample number (primary with body)
    t_tdb          float64    seconds since scenario.epoch (TDB)
    body           string     canonical body / particle name
    x, y, z        float64    position in ICRF barycentric, km
    vx, vy, vz     float64    velocity in ICRF barycentric, km/s
    terminated     bool       True on impact/escape; NaN position thereafter
    energy_rel_err float64    |ΔE/E0| at this sample (same across bodies)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from tomcosmos.state.scenario import Scenario

COLUMNS: tuple[str, ...] = (
    "sample_idx",
    "t_tdb",
    "body",
    "x",
    "y",
    "z",
    "vx",
    "vy",
    "vz",
    "terminated",
    "energy_rel_err",
)


@dataclass(frozen=True)
class StateHistory:
    """Output of a `run()`: a DataFrame + the scenario it came from.

    The DataFrame is the authoritative store. Convenience methods below
    give callers typed access without forcing them to learn the column
    conventions cold.
    """

    df: pd.DataFrame
    scenario: Scenario
    body_names: tuple[str, ...] = field(default_factory=tuple)

    @property
    def n_samples(self) -> int:
        """Number of distinct sample_idx values."""
        if self.df.empty:
            return 0
        return int(self.df["sample_idx"].max()) + 1

    def body_trajectory(self, name: str) -> pd.DataFrame:
        """Slice to one body, return (t_tdb, x, y, z, vx, vy, vz) in order."""
        mask = self.df["body"] == name
        if not mask.any():
            raise KeyError(f"body {name!r} not in StateHistory")
        return (
            self.df.loc[mask, ["t_tdb", "x", "y", "z", "vx", "vy", "vz"]]
            .sort_values("t_tdb")
            .reset_index(drop=True)
        )

    def energy_trace(self) -> pd.DataFrame:
        """Per-sample (sample_idx, t_tdb, energy_rel_err). One row per sample,
        collapsed from the repeated-across-bodies column."""
        return (
            self.df.groupby("sample_idx", as_index=False)
            .agg(t_tdb=("t_tdb", "first"), energy_rel_err=("energy_rel_err", "first"))
            .sort_values("sample_idx")
            .reset_index(drop=True)
        )
