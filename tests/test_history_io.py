"""Parquet I/O + RunMetadata round-trip tests.

Uses an explicit-IC Sun-Earth scenario so these run without the ephemeris
kernel. The full sun+planets roundtrip lives as an ephemeris-marked test.
"""
from __future__ import annotations

import math
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from tomcosmos import RunMetadata, Scenario, load_run, run
from tomcosmos.runner import resolve_output_path
from tomcosmos.state.ephemeris import EphemerisSource

AU_KM = 1.495978707e8


def _explicit_scenario(tmp_path: Path | None = None) -> Scenario:
    return Scenario.model_validate(
        {
            "schema_version": 1,
            "name": "io-test",
            "epoch": "2026-01-01T00:00:00 TDB",
            "duration": "30 day",
            "integrator": {"name": "whfast", "timestep": "1 day"},
            "output": {"format": "parquet", "cadence": "5 day"},
            "bodies": [
                {"name": "sun", "mass_kg": 1.989e30, "radius_km": 695700.0,
                 "ic": {"source": "explicit", "r": [0, 0, 0], "v": [0, 0, 0]}},
                {"name": "earth", "mass_kg": 5.9724e24, "radius_km": 6371.0,
                 "ic": {"source": "explicit",
                        "r": [AU_KM, 0, 0], "v": [0, 29.7847, 0]}},
            ],
        }
    )


class _NoEphemerisNeeded(EphemerisSource):  # type: ignore[misc]
    def __init__(self) -> None:  # skip kernel load
        pass

    def query(self, body, epoch):  # type: ignore[no-untyped-def]
        raise AssertionError("explicit-IC scenario shouldn't need ephemeris")

    def available_bodies(self):  # type: ignore[override]
        return ()

    def time_range(self):  # type: ignore[override]
        from astropy.time import Time
        return Time("1900-01-01", scale="tdb"), Time("2100-01-01", scale="tdb")


# --- RunMetadata capture -----------------------------------------------------


def test_run_captures_metadata() -> None:
    history = run(_explicit_scenario(), source=_NoEphemerisNeeded())
    assert history.metadata is not None
    md = history.metadata
    assert len(md.run_id) == 32  # uuid4 hex
    assert len(md.scenario_sha256) == 64
    assert md.scenario_yaml.startswith("bodies")  # sorted keys
    assert md.schema_version_at_run == 1
    assert md.schema_version_current == 1
    assert md.wallclock_seconds >= 0.0


def test_metadata_captures_git_sha_when_in_repo() -> None:
    history = run(_explicit_scenario(), source=_NoEphemerisNeeded())
    md = history.metadata
    assert md is not None
    # The tomcosmos repo is a git repo, so we should get a SHA.
    assert md.git_sha is not None
    assert len(md.git_sha) == 40
    # dirty may be True during local dev; we don't assert either way.


def test_metadata_captures_package_versions() -> None:
    history = run(_explicit_scenario(), source=_NoEphemerisNeeded())
    md = history.metadata
    assert md is not None
    assert md.rebound_version != "unknown"
    assert md.astropy_version != "unknown"
    assert md.numpy_version != "unknown"
    assert md.pyarrow_version != "unknown"
    assert "." in md.python_version


# --- Parquet write/read ------------------------------------------------------


def test_to_parquet_round_trips_dataframe(tmp_path: Path) -> None:
    history = run(_explicit_scenario(), source=_NoEphemerisNeeded())
    out = tmp_path / "run.parquet"
    history.to_parquet(out)
    assert out.exists()
    loaded = load_run(out)
    assert len(loaded.df) == len(history.df)
    # Column-by-column equality (ignore pandas index differences).
    for col in history.df.columns:
        orig = history.df[col].reset_index(drop=True)
        loaded_col = loaded.df[col].reset_index(drop=True)
        if col == "body":
            assert list(orig.astype(str)) == list(loaded_col.astype(str))
        else:
            assert (orig == loaded_col).all() or orig.equals(loaded_col)


def test_to_parquet_round_trips_scenario(tmp_path: Path) -> None:
    history = run(_explicit_scenario(), source=_NoEphemerisNeeded())
    out = tmp_path / "run.parquet"
    history.to_parquet(out)
    loaded = load_run(out)
    assert loaded.scenario.name == history.scenario.name
    assert loaded.scenario.schema_version == history.scenario.schema_version
    assert len(loaded.scenario.bodies) == len(history.scenario.bodies)
    # Duration should round-trip through canonical YAML.
    assert math.isclose(
        loaded.scenario.duration.value, history.scenario.duration.value, rel_tol=1e-9
    )


def test_to_parquet_round_trips_metadata(tmp_path: Path) -> None:
    history = run(_explicit_scenario(), source=_NoEphemerisNeeded())
    out = tmp_path / "run.parquet"
    history.to_parquet(out)
    loaded = load_run(out)
    assert loaded.metadata is not None
    assert isinstance(loaded.metadata, RunMetadata)
    assert loaded.metadata.run_id == history.metadata.run_id  # type: ignore[union-attr]
    assert loaded.metadata.scenario_sha256 == history.metadata.scenario_sha256  # type: ignore[union-attr]
    assert loaded.metadata.git_sha == history.metadata.git_sha  # type: ignore[union-attr]


def test_body_names_recovered_from_parquet(tmp_path: Path) -> None:
    history = run(_explicit_scenario(), source=_NoEphemerisNeeded())
    out = tmp_path / "run.parquet"
    history.to_parquet(out)
    loaded = load_run(out)
    assert set(loaded.body_names) == {"sun", "earth"}


def test_to_parquet_refuses_overwrite(tmp_path: Path) -> None:
    history = run(_explicit_scenario(), source=_NoEphemerisNeeded())
    out = tmp_path / "run.parquet"
    history.to_parquet(out)
    with pytest.raises(FileExistsError):
        history.to_parquet(out)


def test_to_parquet_overwrite_true(tmp_path: Path) -> None:
    history = run(_explicit_scenario(), source=_NoEphemerisNeeded())
    out = tmp_path / "run.parquet"
    history.to_parquet(out)
    history.to_parquet(out, overwrite=True)  # should not raise


def test_to_parquet_creates_parent_dirs(tmp_path: Path) -> None:
    history = run(_explicit_scenario(), source=_NoEphemerisNeeded())
    out = tmp_path / "deep" / "nested" / "run.parquet"
    history.to_parquet(out)
    assert out.exists()


def test_body_column_is_dictionary_encoded(tmp_path: Path) -> None:
    """Parquet-level: `body` should be dict-encoded for space efficiency."""
    history = run(_explicit_scenario(), source=_NoEphemerisNeeded())
    out = tmp_path / "run.parquet"
    history.to_parquet(out)
    parquet_file = pq.ParquetFile(out)
    # Find the body column's encoding in the first row group.
    row_group = parquet_file.metadata.row_group(0)
    body_col_idx = [
        i for i in range(row_group.num_columns)
        if row_group.column(i).path_in_schema == "body"
    ][0]
    encodings = row_group.column(body_col_idx).encodings
    # RLE_DICTIONARY or PLAIN_DICTIONARY should appear when dict encoding fires.
    assert any("DICTIONARY" in str(e) for e in encodings)


def test_load_run_rejects_non_tomcosmos_parquet(tmp_path: Path) -> None:
    """A plain Parquet file without our metadata key should be rejected."""
    import pandas as pd
    df = pd.DataFrame({"a": [1, 2, 3]})
    p = tmp_path / "plain.parquet"
    df.to_parquet(p)
    with pytest.raises(ValueError, match="tomcosmos_scenario_yaml"):
        load_run(p)


# --- run(write=True) orchestration ------------------------------------------


def test_run_with_write_produces_file(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TOMCOSMOS_RUNS_DIR", str(tmp_path))
    history = run(_explicit_scenario(), source=_NoEphemerisNeeded(), write=True)
    # Default path is runs_dir/<name>__<timestamp>.parquet
    files = list(tmp_path.glob("io-test__*.parquet"))
    assert len(files) == 1
    loaded = load_run(files[0])
    assert loaded.scenario.name == history.scenario.name


def test_run_with_write_explicit_path(tmp_path: Path) -> None:
    scenario = Scenario.model_validate(
        {
            "schema_version": 1,
            "name": "io-test",
            "epoch": "2026-01-01T00:00:00 TDB",
            "duration": "30 day",
            "integrator": {"name": "whfast", "timestep": "1 day"},
            "output": {
                "format": "parquet",
                "cadence": "5 day",
                "path": str(tmp_path / "explicit.parquet"),
            },
            "bodies": [
                {"name": "sun", "mass_kg": 1.989e30, "radius_km": 695700.0,
                 "ic": {"source": "explicit", "r": [0, 0, 0], "v": [0, 0, 0]}},
                {"name": "earth", "mass_kg": 5.9724e24, "radius_km": 6371.0,
                 "ic": {"source": "explicit",
                        "r": [AU_KM, 0, 0], "v": [0, 29.7847, 0]}},
            ],
        }
    )
    run(scenario, source=_NoEphemerisNeeded(), write=True)
    assert (tmp_path / "explicit.parquet").exists()


def test_run_with_write_refuses_existing(tmp_path: Path) -> None:
    (tmp_path / "conflict.parquet").write_bytes(b"preexisting")
    scenario = Scenario.model_validate(
        {
            "schema_version": 1,
            "name": "io-test",
            "epoch": "2026-01-01T00:00:00 TDB",
            "duration": "30 day",
            "integrator": {"name": "whfast", "timestep": "1 day"},
            "output": {
                "format": "parquet",
                "cadence": "5 day",
                "path": str(tmp_path / "conflict.parquet"),
            },
            "bodies": [
                {"name": "sun", "mass_kg": 1.989e30, "radius_km": 695700.0,
                 "ic": {"source": "explicit", "r": [0, 0, 0], "v": [0, 0, 0]}},
                {"name": "earth", "mass_kg": 5.9724e24, "radius_km": 6371.0,
                 "ic": {"source": "explicit",
                        "r": [AU_KM, 0, 0], "v": [0, 29.7847, 0]}},
            ],
        }
    )
    with pytest.raises(FileExistsError):
        run(scenario, source=_NoEphemerisNeeded(), write=True)


def test_resolve_output_path_uses_metadata_timestamp() -> None:
    history = run(_explicit_scenario(), source=_NoEphemerisNeeded())
    assert history.metadata is not None
    path = resolve_output_path(history.scenario, history.metadata)
    assert "io-test__" in path.name
    assert path.suffix == ".parquet"


# --- Dirty-tree gate ---------------------------------------------------------


def test_allow_dirty_true_is_default() -> None:
    """Library default is to allow dirty — CLI flips this to False.
    Verifying we can run without committing every scratch file."""
    history = run(_explicit_scenario(), source=_NoEphemerisNeeded())
    assert history.metadata is not None


# --- Ephemeris-backed end-to-end --------------------------------------------


@pytest.mark.ephemeris
def test_sun_planets_round_trip_through_parquet(
    ephemeris_source: EphemerisSource, tmp_path: Path,
) -> None:
    scenario = Scenario.model_validate(
        {
            "schema_version": 1,
            "name": "sun-planets-io",
            "epoch": "2026-04-23T00:00:00 TDB",
            "duration": "30 day",
            "integrator": {"name": "whfast", "timestep": "1 day"},
            "output": {"format": "parquet", "cadence": "5 day"},
            "bodies": [
                {"name": n, "spice_id": sid, "ic": {"source": "ephemeris"}}
                for n, sid in [
                    ("sun", 10), ("mercury", 199), ("venus", 299),
                    ("earth", 399), ("mars", 499), ("jupiter", 599),
                    ("saturn", 699), ("uranus", 799), ("neptune", 899),
                ]
            ],
        }
    )
    history = run(scenario, source=ephemeris_source)
    out = tmp_path / "sun-planets.parquet"
    history.to_parquet(out)
    loaded = load_run(out)
    assert loaded.metadata is not None
    # kernel_hashes populated from the real EphemerisSource
    assert "de440s.bsp" in loaded.metadata.kernel_hashes
    assert len(loaded.metadata.kernel_hashes["de440s.bsp"]) == 64
    # DataFrame same length
    assert len(loaded.df) == len(history.df) == 7 * 9
