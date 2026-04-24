"""Run metadata capture — reproducibility + diagnostics.

Every `run()` invocation produces a `RunMetadata` that travels with the
Parquet output (embedded in file-level key/value metadata). The goal is
that someone with the Parquet file alone can:
  - re-run the same scenario (scenario_yaml is embedded),
  - pin the exact code that produced it (git_sha),
  - detect silent kernel swaps (kernel_hashes),
  - attribute performance numbers (wallclock_seconds).

Structured logging (PLAN.md > "Canonical log-event fields") lands later
— for now we just build the metadata block.
"""
from __future__ import annotations

import hashlib
import platform as _platform
import subprocess
import sys
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from tomcosmos.exceptions import DirtyWorkingTreeError
from tomcosmos.state.ephemeris import EphemerisSource
from tomcosmos.state.scenario import SCHEMA_VERSION, Scenario


@dataclass(frozen=True)
class RunMetadata:
    """Everything needed to re-derive or re-run this trajectory.

    All fields are plain JSON-serializable types so the whole struct
    encodes cleanly into Parquet file metadata via `asdict + json.dumps`.
    """

    run_id: str
    scenario_sha256: str
    scenario_yaml: str
    schema_version_at_run: int
    schema_version_current: int
    git_sha: str | None
    git_dirty: bool
    rebound_version: str
    astropy_version: str
    numpy_version: str
    pyarrow_version: str
    python_version: str
    platform: str
    kernel_hashes: dict[str, str] = field(default_factory=dict)
    kernel_versions: dict[str, str] = field(default_factory=dict)
    start_wallclock: str = ""
    end_wallclock: str = ""
    wallclock_seconds: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> RunMetadata:
        # asdict nests dicts fine; reconstruct with a cast for the float field
        # that JSON may have serialized as int.
        d = dict(d)
        raw_wall = d.get("wallclock_seconds", 0.0)
        d["wallclock_seconds"] = float(raw_wall)  # type: ignore[arg-type]
        return cls(**d)  # type: ignore[arg-type]


def capture_metadata(
    scenario: Scenario,
    source: EphemerisSource,
    start: datetime,
    end: datetime,
    *,
    repo_root: Path | None = None,
    allow_dirty: bool = False,
) -> RunMetadata:
    """Build a `RunMetadata` from the scenario, ephemeris source, and timing.

    If the working tree is dirty and `allow_dirty=False`, raises
    `DirtyWorkingTreeError`. This is the gate that keeps "I ran some
    uncommitted experiment" from silently shipping as a clean result.
    """
    root = repo_root if repo_root is not None else _find_repo_root()
    sha, dirty = _git_state(root)
    if dirty and not allow_dirty:
        raise DirtyWorkingTreeError(
            "working tree has uncommitted changes; commit, stash, or "
            "pass allow_dirty=True (CLI: --allow-dirty)"
        )

    canonical_yaml = scenario.to_yaml_string()
    scenario_sha = hashlib.sha256(canonical_yaml.encode("utf-8")).hexdigest()

    return RunMetadata(
        run_id=uuid.uuid4().hex,
        scenario_sha256=scenario_sha,
        scenario_yaml=canonical_yaml,
        schema_version_at_run=int(scenario.schema_version),
        schema_version_current=int(SCHEMA_VERSION),
        git_sha=sha,
        git_dirty=dirty,
        rebound_version=_pkg_version("rebound"),
        astropy_version=_pkg_version("astropy"),
        numpy_version=_pkg_version("numpy"),
        pyarrow_version=_pkg_version("pyarrow"),
        python_version=sys.version.split()[0],
        platform=_platform.platform(),
        kernel_hashes=_kernel_hashes(source),
        kernel_versions={},  # M2: extract from SPICE COMMENT area
        start_wallclock=_iso(start),
        end_wallclock=_iso(end),
        wallclock_seconds=(end - start).total_seconds(),
    )


def _iso(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat(timespec="microseconds")


def _pkg_version(name: str) -> str:
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:  # pragma: no cover — Python <3.8 only
        return "unknown"
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def _find_repo_root(start: Path | None = None) -> Path | None:
    """Walk up looking for a .git directory; returns None if not a git repo."""
    here = Path(start) if start is not None else Path.cwd()
    for candidate in (here, *here.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def _git_state(repo_root: Path | None) -> tuple[str | None, bool]:
    """(HEAD_sha_or_None, is_dirty). Returns (None, False) outside a git repo."""
    if repo_root is None:
        return None, False
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root, capture_output=True, text=True, check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root, capture_output=True, text=True, check=True,
        ).stdout
        return sha, bool(status.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None, False


def _kernel_hashes(source: EphemerisSource) -> dict[str, str]:
    """SHA256 of every kernel file the source has loaded.

    SkyfieldSource exposes `kernel_path`; other backends grow the same
    or a `kernel_paths()` method when M2 needs multiple files. Accessing
    `kernel_path` goes through a property, which can raise on test
    stubs that don't fully initialize — swallow that and report empty.
    """
    paths: list[Path] = []
    try:
        maybe_path = source.kernel_path  # type: ignore[attr-defined]
    except AttributeError:
        maybe_path = None
    if isinstance(maybe_path, Path) and maybe_path.exists():
        paths.append(maybe_path)
    return {p.name: _sha256_file(p) for p in paths}


def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()
