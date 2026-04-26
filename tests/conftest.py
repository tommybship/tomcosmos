"""Shared pytest fixtures.

Session-scoped where possible — loading kernels per test is a few hundred
ms each and adds up. `tmp_runs_dir` isolates per-test output so parallel
runs don't collide.

`ephemeris_source` is the single skyfield-backed source over
`data/kernels/de440s.bsp` — used for IC seeding in any test that
constructs a Mode B simulation. Tests that exercise Mode A (ASSIST)
don't touch this fixture; ASSIST reads its own kernels in the force
loop.
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
def ephemeris_source(kernel_dir: Path) -> Iterator[object]:
    """Session-scoped `EphemerisSource` over `data/kernels/de440s.bsp`.

    Skipped if the kernel hasn't been fetched yet; run `scripts/hello_world.py`
    or `python -m tomcosmos.cli fetch-kernels` first.
    """
    kernel_path = kernel_dir / "de440s.bsp"
    if not kernel_path.exists():
        pytest.skip(f"kernel not present at {kernel_path}; run fetch-kernels first")
    from tomcosmos.state.ephemeris import EphemerisSource

    src = EphemerisSource(kernel_filename="de440s.bsp", directory=kernel_dir)
    try:
        yield src
    finally:
        src.close()


@pytest.fixture
def tmp_runs_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Redirect TOMCOSMOS_RUNS_DIR to a per-test tmp path."""
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setenv("TOMCOSMOS_RUNS_DIR", str(runs))
    yield runs
