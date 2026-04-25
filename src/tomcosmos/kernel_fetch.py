"""Kernel downloader + manifest writer.

Implementation lives here (reusable from CLI and `scripts/fetch_kernels.py`),
with a thin CLI wrapper in `scripts/`. Each fetched kernel:
  - Lands in `tomcosmos.config.kernel_dir()`.
  - Is hashed with SHA256 after download.
  - Gets an entry written to `<kernel_dir>/manifest.json`:
      {"de440s.bsp": {"url": ..., "sha256": ..., "downloaded_at": ...}, ...}

The ephemeris loader reads this manifest to pin which exact kernel
revision produced a given run — see PLAN.md > "Kernel locking."
"""
from __future__ import annotations

import hashlib
import json
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tomcosmos.config import kernel_dir
from tomcosmos.kernels import KernelGroup


def fetch_groups(
    groups: list[KernelGroup],
    directory: Path | None = None,
    *,
    upgrade: bool = False,
) -> None:
    """Download every group's kernel if missing, then update the manifest.

    Skips downloads when the file is already on disk. Validates SHA256
    against the manifest entry if present (silent kernel swaps would
    otherwise be invisible).

    With `upgrade=True`, deletes any *.bsp in the kernel directory that
    isn't named by any group in the registry — orphaned files left behind
    when we bump a kernel version (e.g., sat441 → sat459).
    """
    d = directory if directory is not None else kernel_dir()
    d.mkdir(parents=True, exist_ok=True)
    manifest_path = d / "manifest.json"
    manifest = _read_manifest(manifest_path)

    total_bytes = 0
    for g in groups:
        target = d / g.filename
        if target.exists():
            print(f"  {g.filename}: already present ({target.stat().st_size / 1e6:.1f} MB)")
            # Verify hash matches what manifest claims, if any.
            existing_sha = manifest.get(g.filename, {}).get("sha256")
            if existing_sha:
                actual = _sha256(target)
                if actual != existing_sha:
                    print(
                        f"  WARNING: {g.filename} on disk ({actual[:12]}...) "
                        f"differs from manifest ({existing_sha[:12]}...)."
                    )
            continue

        size_mb = g.approx_size_mb
        print(f"  {g.filename}: downloading ~{size_mb:.0f} MB from NAIF...")
        _download(g.url, target)
        sha = _sha256(target)
        actual_mb = target.stat().st_size / 1e6
        manifest[g.filename] = {
            "url": g.url,
            "sha256": sha,
            "downloaded_at": datetime.now(UTC).isoformat(),
            "size_bytes": target.stat().st_size,
            "group": g.name,
        }
        total_bytes += target.stat().st_size
        print(f"  {g.filename}: {actual_mb:.1f} MB, sha256 {sha[:12]}...")

    if upgrade:
        _prune_orphans(d, manifest)

    _write_manifest(manifest_path, manifest)
    if total_bytes:
        print(f"done. wrote {total_bytes / 1e6:.1f} MB to {d}")
    else:
        print(f"done. nothing to fetch; {d} already has every requested kernel.")


def _prune_orphans(d: Path, manifest: dict[str, dict[str, Any]]) -> None:
    """Delete .bsp files in `d` whose names aren't in any current registry
    group. Updates the manifest in place to drop the stale entries.

    Called only with --upgrade so a normal fetch never deletes anything.
    """
    from tomcosmos.kernels import ALL_GROUPS

    valid_filenames = {g.filename for g in ALL_GROUPS}
    freed = 0
    for path in sorted(d.glob("*.bsp")):
        if path.name in valid_filenames:
            continue
        size = path.stat().st_size
        path.unlink()
        manifest.pop(path.name, None)
        freed += size
        print(f"  pruned orphan: {path.name} ({size / 1e6:.1f} MB)")
    if freed:
        print(f"  freed {freed / 1e6:.1f} MB")


def _download(url: str, dest: Path) -> None:
    """Stream-download `url` to `dest`. Atomic via tmpfile + rename."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as f:
        while True:
            chunk = resp.read(1 << 20)  # 1 MB chunks
            if not chunk:
                break
            f.write(chunk)
    tmp.replace(dest)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_manifest(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
        return {}
    except json.JSONDecodeError:
        return {}


def _write_manifest(path: Path, manifest: dict[str, dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
