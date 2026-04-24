"""CLI tests via typer's CliRunner (no subprocesses)."""
from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from tomcosmos.cli import app

AU_KM = 1.495978707e8
runner = CliRunner()


FIXTURES = Path(__file__).parent / "fixtures" / "scenarios"


# --- version -----------------------------------------------------------------


def test_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "tomcosmos" in result.stdout


# --- validate ----------------------------------------------------------------


@pytest.mark.ephemeris
def test_validate_good_scenario() -> None:
    result = runner.invoke(app, ["validate", str(FIXTURES / "good_sun_planets.yaml")])
    assert result.exit_code == 0
    assert "OK" in result.stdout
    assert "schema_version: 1" in result.stdout
    assert "bodies: 9" in result.stdout


def test_validate_missing_schema_version(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "name: foo\n"
        "epoch: '2026-01-01T00:00:00 TDB'\n"
        "duration: '1 yr'\n"
        "integrator: {name: whfast, timestep: '1 day'}\n"
        "output: {format: parquet, cadence: '1 day'}\n"
        "bodies: [{name: sun, spice_id: 10, ic: {source: ephemeris}}]\n"
    )
    result = runner.invoke(app, ["validate", str(bad)])
    assert result.exit_code == 2
    assert "schema_version" in result.stderr


def test_validate_unknown_integrator(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "schema_version: 1\n"
        "name: foo\n"
        "epoch: '2026-01-01T00:00:00 TDB'\n"
        "duration: '1 yr'\n"
        "integrator: {name: leapfrog}\n"
        "output: {format: parquet, cadence: '1 day'}\n"
        "bodies: [{name: sun, spice_id: 10, ic: {source: ephemeris}}]\n"
    )
    result = runner.invoke(app, ["validate", str(bad)])
    assert result.exit_code == 2


# --- run ---------------------------------------------------------------------


def _write_explicit_scenario(path: Path, name: str = "cli-test") -> Path:
    path.write_text(
        f"schema_version: 1\n"
        f"name: {name}\n"
        f"epoch: '2026-01-01T00:00:00 TDB'\n"
        f"duration: '30 day'\n"
        f"integrator: {{name: whfast, timestep: '1 day'}}\n"
        f"output: {{format: parquet, cadence: '5 day'}}\n"
        f"bodies:\n"
        f"  - name: sun\n"
        f"    mass_kg: 1.989e30\n"
        f"    radius_km: 695700.0\n"
        f"    ic: {{source: explicit, r: [0, 0, 0], v: [0, 0, 0]}}\n"
        f"  - name: earth\n"
        f"    mass_kg: 5.9724e24\n"
        f"    radius_km: 6371.0\n"
        f"    ic: {{source: explicit, r: [{AU_KM}, 0, 0], v: [0, 29.7847, 0]}}\n"
    )
    return path


def test_run_writes_parquet_and_prints_summary(tmp_path: Path) -> None:
    scenario = _write_explicit_scenario(tmp_path / "s.yaml")
    out = tmp_path / "out.parquet"
    result = runner.invoke(
        app,
        ["run", str(scenario), "--output", str(out), "--allow-dirty"],
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    assert out.exists()
    assert "wrote" in result.stdout
    assert "run_id" in result.stdout
    assert "samples" in result.stdout
    assert "max |dE/E|" in result.stdout


def test_run_refuses_existing_output(tmp_path: Path) -> None:
    scenario = _write_explicit_scenario(tmp_path / "s.yaml")
    out = tmp_path / "out.parquet"
    out.write_bytes(b"preexisting")
    result = runner.invoke(
        app,
        ["run", str(scenario), "--output", str(out), "--allow-dirty"],
    )
    assert result.exit_code == 5
    assert "--overwrite" in result.stderr


def test_run_overwrite_flag_allows_replacement(tmp_path: Path) -> None:
    scenario = _write_explicit_scenario(tmp_path / "s.yaml")
    out = tmp_path / "out.parquet"
    # First run
    r1 = runner.invoke(
        app, ["run", str(scenario), "--output", str(out), "--allow-dirty"]
    )
    assert r1.exit_code == 0
    # Second run with --overwrite
    r2 = runner.invoke(
        app,
        ["run", str(scenario), "--output", str(out),
         "--allow-dirty", "--overwrite"],
    )
    assert r2.exit_code == 0


def test_run_bad_scenario_exits_code_2(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("schema_version: 999\nname: x\n")  # bogus schema version
    result = runner.invoke(app, ["run", str(bad), "--allow-dirty"])
    assert result.exit_code == 2


def test_run_nonexistent_scenario_exits_nonzero(tmp_path: Path) -> None:
    # typer's exists=True Argument validation catches this
    result = runner.invoke(app, ["run", str(tmp_path / "missing.yaml")])
    assert result.exit_code != 0


# --- info --------------------------------------------------------------------


def test_info_prints_metadata(tmp_path: Path) -> None:
    scenario = _write_explicit_scenario(tmp_path / "s.yaml")
    out = tmp_path / "out.parquet"
    runner.invoke(
        app, ["run", str(scenario), "--output", str(out), "--allow-dirty"]
    )
    result = runner.invoke(app, ["info", str(out)])
    assert result.exit_code == 0
    assert "run_id:" in result.stdout
    assert "git_sha:" in result.stdout
    assert "versions: rebound=" in result.stdout
    assert "platform:" in result.stdout


def test_info_rejects_non_tomcosmos_parquet(tmp_path: Path) -> None:
    import pandas as pd
    p = tmp_path / "plain.parquet"
    pd.DataFrame({"a": [1, 2, 3]}).to_parquet(p)
    result = runner.invoke(app, ["info", str(p)])
    assert result.exit_code == 5
    assert "tomcosmos_scenario_yaml" in result.stderr
