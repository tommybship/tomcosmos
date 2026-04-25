"""Shared pytest fixtures.

Session-scoped where possible — loading kernels per test is a few hundred
ms each and adds up. `tmp_runs_dir` isolates per-test output so parallel
runs don't collide.

Ephemeris fixtures: `skyfield_source` and `spice_source` provide one
backend each; `ephemeris_source` is parametrized over both so contract
tests run against the union. Use the parametrized version unless the
test is exercising backend-specific behavior (kernel paths, error
messages, etc.).
"""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KERNEL_DIR = REPO_ROOT / "data" / "kernels"


@pytest.fixture(scope="session")
def kernel_dir() -> Path:
    """The repo's kernel directory; skip tests that need it if the dir is empty."""
    return DEFAULT_KERNEL_DIR


@pytest.fixture(scope="session")
def skyfield_source(kernel_dir: Path) -> Iterator[object]:
    """Session-scoped SkyfieldSource over `data/kernels/de440s.bsp`.

    Skipped if the kernel hasn't been fetched yet; run `scripts/hello_world.py`
    or `python -m tomcosmos.cli fetch-kernels` first.
    """
    kernel_path = kernel_dir / "de440s.bsp"
    if not kernel_path.exists():
        pytest.skip(f"kernel not present at {kernel_path}; run fetch-kernels first")
    from tomcosmos.state.ephemeris import SkyfieldSource

    src = SkyfieldSource(kernel_filename="de440s.bsp", directory=kernel_dir)
    try:
        yield src
    finally:
        src.close()


@pytest.fixture(scope="session")
def spice_source(kernel_dir: Path) -> Iterator[object]:
    """Session-scoped SpiceSource loading every .bsp in the kernel dir.

    SpiceSource furnshes into the *process-global* spiceypy kernel pool;
    the refcount in ephemeris._SPICE_REFS keeps multiple instances safe,
    and the fixture's teardown calls close() so the pool drains at end
    of session.
    """
    kernel_path = kernel_dir / "de440s.bsp"
    if not kernel_path.exists():
        pytest.skip(f"kernel not present at {kernel_path}; run fetch-kernels first")
    from tomcosmos.state.ephemeris import SpiceSource

    src = SpiceSource(directory=kernel_dir)
    try:
        yield src
    finally:
        src.close()


@pytest.fixture(scope="session", params=["skyfield", "spice"])
def ephemeris_source(request: pytest.FixtureRequest) -> object:
    """Parametrized over both ephemeris backends. Use for contract tests."""
    return request.getfixturevalue(f"{request.param}_source")


@pytest.fixture
def tmp_runs_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Redirect TOMCOSMOS_RUNS_DIR to a per-test tmp path."""
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setenv("TOMCOSMOS_RUNS_DIR", str(runs))
    yield runs
