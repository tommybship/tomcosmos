"""StateHistory — in-memory representation of a simulation's output.

Long-format pandas DataFrame: one row per (sample, body), plus a
`RunMetadata` block that travels with the data when written to
Parquet.

Schema (see PLAN.md > "StateHistory schema"):
    sample_idx     int64      0-indexed sample number (primary with body)
    t_tdb          float64    seconds since scenario.epoch (TDB)
    body           string     canonical body / particle name
    x, y, z        float64    position in ICRF barycentric, km
    vx, vy, vz     float64    velocity in ICRF barycentric, km/s
    terminated     bool       True on impact/escape; NaN position thereafter
    energy_rel_err float64    |ΔE/E0| at this sample (same across bodies)

File metadata (Parquet key/value):
    tomcosmos_run_metadata   JSON-encoded RunMetadata.to_dict()
    tomcosmos_scenario_yaml  canonical scenario YAML (also inside metadata
                             — lifted out to a dedicated key for cheap
                             inspection via `pyarrow.parquet.read_schema`).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from tomcosmos.io.diagnostics import RunMetadata
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

_META_KEY = b"tomcosmos_run_metadata"
_SCENARIO_KEY = b"tomcosmos_scenario_yaml"


@dataclass(frozen=True)
class StateHistory:
    """Output of `run()` — DataFrame + scenario + run metadata.

    `metadata` is optional so test helpers can construct histories by hand,
    but any history produced by `run()` will have it populated. Writing
    to Parquet without metadata is allowed but emits a minimal file.

    `events` is the per-sample encounter / Δv / impact log. Empty when
    the run has no test particles or no detected events. Persists as a
    sidecar parquet (`<basename>.events.parquet`) next to the trajectory
    file — see `to_parquet` and `from_parquet`.
    """

    df: pd.DataFrame
    scenario: Scenario
    body_names: tuple[str, ...] = field(default_factory=tuple)
    metadata: RunMetadata | None = None
    events: pd.DataFrame | None = None

    # --- Accessors ------------------------------------------------------

    @property
    def n_samples(self) -> int:
        if self.df.empty:
            return 0
        return int(self.df["sample_idx"].max()) + 1

    def body_trajectory(self, name: str) -> pd.DataFrame:
        mask = self.df["body"] == name
        if not mask.any():
            raise KeyError(f"body {name!r} not in StateHistory")
        return (
            self.df.loc[mask, ["t_tdb", "x", "y", "z", "vx", "vy", "vz"]]
            .sort_values("t_tdb")
            .reset_index(drop=True)
        )

    def energy_trace(self) -> pd.DataFrame:
        return (
            self.df.groupby("sample_idx", as_index=False)
            .agg(t_tdb=("t_tdb", "first"), energy_rel_err=("energy_rel_err", "first"))
            .sort_values("sample_idx")
            .reset_index(drop=True)
        )

    # --- Parquet I/O ----------------------------------------------------

    def to_parquet(self, path: str | Path, *, overwrite: bool = False) -> Path:
        """Write to `path`. Creates parent dirs. Refuses to overwrite an
        existing file unless `overwrite=True`.

        Events (when present and non-empty) are written to a sidecar
        parquet at `<path stem>.events.parquet` so big trajectory files
        don't bloat with sparse event data. The sidecar is overwritten
        in lockstep with the trajectory whenever `overwrite=True`.
        """
        p = Path(path)
        if p.exists() and not overwrite:
            raise FileExistsError(
                f"{p} exists; pass overwrite=True (CLI: --overwrite) to replace"
            )
        p.parent.mkdir(parents=True, exist_ok=True)

        table = pa.Table.from_pandas(self.df, preserve_index=False)
        file_md: dict[bytes, bytes] = {
            _SCENARIO_KEY: self.scenario.to_yaml_string().encode("utf-8"),
        }
        if self.metadata is not None:
            file_md[_META_KEY] = json.dumps(self.metadata.to_dict()).encode("utf-8")

        existing = table.schema.metadata or {}
        merged = {**existing, **file_md}
        table = table.replace_schema_metadata(merged)

        pq.write_table(
            table, p,
            use_dictionary=["body"],  # small integer codes + shared name pool
            compression="snappy",
        )

        events_path = _events_sidecar_path(p)
        if self.events is not None and not self.events.empty:
            events_table = pa.Table.from_pandas(self.events, preserve_index=False)
            pq.write_table(events_table, events_path, compression="snappy")
        elif events_path.exists() and overwrite:
            # Stale sidecar from a prior run that had events — clear it so
            # the on-disk pair matches the in-memory state.
            events_path.unlink()
        return p

    @classmethod
    def from_parquet(cls, path: str | Path) -> StateHistory:
        """Inverse of `to_parquet`. Reconstructs `Scenario` from embedded YAML
        and `RunMetadata` from the JSON metadata key (if present)."""
        p = Path(path)
        table = pq.read_table(p)
        md_raw = table.schema.metadata or {}

        scenario_yaml_bytes = md_raw.get(_SCENARIO_KEY)
        if scenario_yaml_bytes is None:
            raise ValueError(
                f"{p}: missing 'tomcosmos_scenario_yaml' metadata — "
                "not a tomcosmos run output?"
            )
        scenario = Scenario.from_yaml_string(scenario_yaml_bytes.decode("utf-8"))

        metadata: RunMetadata | None = None
        meta_bytes = md_raw.get(_META_KEY)
        if meta_bytes is not None:
            metadata = RunMetadata.from_dict(json.loads(meta_bytes.decode("utf-8")))

        df = table.to_pandas()
        body_names = tuple(df["body"].astype(str).unique().tolist())

        events: pd.DataFrame | None = None
        events_path = _events_sidecar_path(p)
        if events_path.exists():
            events = pq.read_table(events_path).to_pandas()

        return cls(
            df=df, scenario=scenario, body_names=body_names,
            metadata=metadata, events=events,
        )


def _events_sidecar_path(trajectory_path: Path) -> Path:
    """`runs/foo.parquet` -> `runs/foo.events.parquet`."""
    return trajectory_path.with_suffix(".events.parquet")


def load_run(path: str | Path) -> StateHistory:
    """Public top-level helper — see `StateHistory.from_parquet`."""
    return StateHistory.from_parquet(path)
