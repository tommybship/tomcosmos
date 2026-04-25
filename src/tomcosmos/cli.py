"""tomcosmos CLI.

Thin wrappers around the library API (tomcosmos.run, StateHistory.from_parquet,
Scenario.from_yaml, etc.). Business logic stays out of this module — the CLI's
only jobs are argument parsing, user-friendly error formatting, and mapping
exceptions to exit codes (PLAN.md > "CLI exit codes and progress"):

    0  success
    1  unexpected (typer default)
    2  scenario validation / preflight failure
    3  ephemeris or kernel error
    4  integrator divergence
    5  I/O error
"""
from __future__ import annotations

from pathlib import Path

import typer

from tomcosmos import __version__
from tomcosmos.config import kernel_dir
from tomcosmos.exceptions import (
    DirtyWorkingTreeError,
    EphemerisOutOfRangeError,
    IntegratorDivergedError,
    KernelDriftError,
    ScenarioValidationError,
    UnknownBodyError,
)
from tomcosmos.io.history import StateHistory
from tomcosmos.runner import resolve_output_path, run
from tomcosmos.state.ephemeris import SkyfieldSource
from tomcosmos.state.ic import resolve_scenario
from tomcosmos.state.scenario import Scenario

app = typer.Typer(
    help="Solar system state simulator.",
    no_args_is_help=True,
    add_completion=False,
)


def _error(msg: str, exit_code: int) -> None:
    typer.echo(f"error: {msg}", err=True)
    raise typer.Exit(code=exit_code)


@app.command()
def version() -> None:
    """Print the installed tomcosmos version."""
    typer.echo(f"tomcosmos {__version__}")


@app.command(name="run")
def run_cmd(
    scenario_path: Path = typer.Argument(
        ..., exists=True, dir_okay=False, readable=True,
        metavar="SCENARIO", help="Path to scenario YAML file.",
    ),
    output: Path | None = typer.Option(
        None, "--output", "-o",
        help="Override the scenario's output path.",
    ),
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Allow overwriting an existing output file.",
    ),
    allow_dirty: bool = typer.Option(
        False, "--allow-dirty",
        help="Allow running with uncommitted changes in the git working tree.",
    ),
) -> None:
    """Integrate a scenario and write the resulting Parquet."""
    try:
        scenario = Scenario.from_yaml(scenario_path)
    except ScenarioValidationError as e:
        _error(str(e), exit_code=2)

    try:
        history = run(scenario, write=False, allow_dirty=allow_dirty)
    except DirtyWorkingTreeError as e:
        _error(str(e), exit_code=2)
    except EphemerisOutOfRangeError as e:
        _error(str(e), exit_code=3)
    except KernelDriftError as e:
        _error(str(e), exit_code=3)
    except UnknownBodyError as e:
        _error(str(e), exit_code=3)
    except IntegratorDivergedError as e:
        _error(str(e), exit_code=4)

    assert history.metadata is not None  # run() always captures
    out_path = output if output is not None else resolve_output_path(
        scenario, history.metadata
    )
    try:
        written = history.to_parquet(out_path, overwrite=overwrite)
    except FileExistsError as e:
        _error(
            f"{e}\n(pass --overwrite to replace)",
            exit_code=5,
        )
    except OSError as e:
        _error(str(e), exit_code=5)

    typer.echo(f"wrote {written}")
    typer.echo(f"  run_id: {history.metadata.run_id}")
    typer.echo(f"  samples: {history.n_samples}")
    max_err = history.df["energy_rel_err"].max()
    typer.echo(f"  max |dE/E|: {max_err:.3e}")
    typer.echo(f"  wallclock: {history.metadata.wallclock_seconds:.2f} s")


@app.command(name="validate")
def validate_cmd(
    scenario_path: Path = typer.Argument(
        ..., exists=True, dir_okay=False, readable=True,
        metavar="SCENARIO", help="Path to scenario YAML file.",
    ),
) -> None:
    """Full preflight: schema + ephemeris coverage + body + IC resolution."""
    try:
        scenario = Scenario.from_yaml(scenario_path)
    except ScenarioValidationError as e:
        _error(str(e), exit_code=2)

    try:
        source = SkyfieldSource()
    except Exception as e:  # noqa: BLE001 — skyfield/OS errors
        _error(f"failed to load ephemeris source: {e}", exit_code=3)

    try:
        source.require_covers(scenario.epoch, scenario.duration)
    except EphemerisOutOfRangeError as e:
        _error(str(e), exit_code=3)

    try:
        resolve_scenario(scenario, source)
    except (UnknownBodyError, ScenarioValidationError) as e:
        _error(str(e), exit_code=2)

    typer.echo(f"{scenario_path}: OK")
    typer.echo(f"  schema_version: {scenario.schema_version}")
    typer.echo(f"  name: {scenario.name}")
    typer.echo(f"  epoch: {scenario.epoch.isot} {scenario.epoch.scale.upper()}")
    typer.echo(f"  duration: {scenario.duration}")
    typer.echo(f"  integrator: {scenario.integrator.name}")
    typer.echo(f"  bodies: {len(scenario.bodies)}")
    typer.echo(f"  test_particles: {len(scenario.test_particles)}")


@app.command(name="info")
def info_cmd(
    parquet_path: Path = typer.Argument(
        ..., exists=True, dir_okay=False, readable=True,
        metavar="PARQUET", help="Path to a run's Parquet file.",
    ),
) -> None:
    """Print the metadata embedded in a run output."""
    try:
        history = StateHistory.from_parquet(parquet_path)
    except OSError as e:
        _error(str(e), exit_code=5)
    except ValueError as e:
        _error(str(e), exit_code=5)

    md = history.metadata
    s = history.scenario
    typer.echo(f"file: {parquet_path}")
    typer.echo(f"scenario: {s.name} (schema v{s.schema_version})")
    typer.echo(f"  epoch: {s.epoch.isot} {s.epoch.scale.upper()}")
    typer.echo(f"  duration: {s.duration}")
    typer.echo(f"  integrator: {s.integrator.name}")
    typer.echo(f"  bodies: {len(s.bodies)}")
    typer.echo(f"samples: {history.n_samples}")
    typer.echo(f"rows: {len(history.df)}")
    max_err = history.df["energy_rel_err"].max()
    typer.echo(f"max |dE/E|: {max_err:.3e}")
    if md is None:
        typer.echo("metadata: <none - constructed outside run()>")
        return
    typer.echo(f"run_id: {md.run_id}")
    dirty_marker = " (dirty)" if md.git_dirty else ""
    typer.echo(f"git_sha: {md.git_sha}{dirty_marker}")
    typer.echo(
        f"versions: rebound={md.rebound_version} astropy={md.astropy_version} "
        f"numpy={md.numpy_version} python={md.python_version}"
    )
    typer.echo(f"platform: {md.platform}")
    typer.echo(f"wallclock: {md.wallclock_seconds:.2f} s "
               f"({md.start_wallclock} -> {md.end_wallclock})")
    if md.kernel_hashes:
        typer.echo("kernels:")
        for name, sha in md.kernel_hashes.items():
            typer.echo(f"  {name}: {sha[:12]}...")


@app.command(name="view")
def view_cmd(
    parquet_path: Path = typer.Argument(
        ..., exists=True, dir_okay=False, readable=True,
        metavar="PARQUET", help="Path to a run's Parquet file.",
    ),
    scaling: str = typer.Option(
        "log", "--scaling",
        help="Body scaling: 'log' (default), 'true', or 'marker'.",
    ),
    follow: str | None = typer.Option(
        None, "--follow",
        help="Body name to keep centered in the viewport while time scrubs. "
             "Useful for watching moons orbit their primary (e.g., --follow jupiter).",
    ),
) -> None:
    """Open a 3D viewer on a run's Parquet file."""
    from tomcosmos.viz.pyvista_viewer import Viewer

    try:
        history = StateHistory.from_parquet(parquet_path)
    except (OSError, ValueError) as e:
        _error(str(e), exit_code=5)

    try:
        viewer = Viewer(history, scaling=scaling, follow=follow)  # type: ignore[arg-type]
    except ValueError as e:
        _error(str(e), exit_code=2)
    viewer.show()


@app.command(name="fetch-kernels")
def fetch_kernels_cmd(
    include: list[str] = typer.Option(
        [], "--include", "-i",
        help="Add a kernel group: mars | jupiter | saturn | neptune | pluto. "
             "Default (no flags) fetches just DE440s (~32 MB). Repeatable.",
    ),
    fetch_all: bool = typer.Option(
        False, "--all",
        help="Fetch every group in the registry (base + all satellite kernels). "
             "Currently ~3.4 GB total.",
    ),
    upgrade: bool = typer.Option(
        False, "--upgrade",
        help="After fetching, delete any kernel files in the kernel directory "
             "that aren't named by any current registry group (e.g., sat441 "
             "after we've moved to sat459). Manifest entries pruned alongside.",
    ),
) -> None:
    """Download NAIF kernels into the configured kernel directory.

    Default fetches DE440s (Sun + 8 planets + Moon, ~32 MB). Use
    --include to add satellite groups (each is tens of MB to ~1 GB
    — sizes printed before downloading). Existing files are skipped;
    SHA256 of every kernel is recorded in manifest.json for
    reproducibility.

    --all is a shortcut for "every group in the registry."

    --upgrade prunes stale kernel files left behind when we bump
    a kernel version (e.g., sat441 → sat459) — only removes files
    that aren't named by any current registry group.
    """
    from tomcosmos.kernel_fetch import fetch_groups
    from tomcosmos.kernels import (
        ALL_GROUPS,
        BASE_GROUP,
        SATELLITE_GROUPS,
        group_by_name,
    )

    target = kernel_dir()
    target.mkdir(parents=True, exist_ok=True)
    typer.echo(f"ensuring kernels in {target.resolve()}...")

    if fetch_all:
        groups = list(ALL_GROUPS)
    else:
        groups = [BASE_GROUP]
        for name in include:
            # Backward compatibility: --include all-moons / all still works.
            if name in ("all-moons", "all"):
                groups = list(ALL_GROUPS)
                break
            try:
                groups.append(group_by_name(name))
            except KeyError:
                _error(
                    f"unknown kernel group {name!r}. "
                    f"Known: {', '.join(g.name for g in SATELLITE_GROUPS)} "
                    "(or use --all for every group)",
                    exit_code=2,
                )

    try:
        fetch_groups(groups, directory=target, upgrade=upgrade)
    except Exception as e:  # noqa: BLE001 — network/OS
        _error(f"kernel fetch failed: {e}", exit_code=3)


def main() -> None:  # entry point for tomcosmos script
    app()


if __name__ == "__main__":
    main()
