"""Kernel registry + fetcher tests."""
from __future__ import annotations

from pathlib import Path

import pytest

from tomcosmos.kernel_fetch import _read_manifest, _sha256, _write_manifest
from tomcosmos.kernels import (
    ALL_GROUPS,
    BASE_GROUP,
    SATELLITE_GROUPS,
    group_by_name,
    group_for_body,
)

# --- registry shape ---------------------------------------------------------


def test_base_group_is_de440s() -> None:
    assert BASE_GROUP.filename == "de440s.bsp"
    assert "earth" in BASE_GROUP.bodies
    assert "moon" in BASE_GROUP.bodies


def test_satellite_groups_cover_galileans() -> None:
    jupiter = group_by_name("jupiter")
    assert jupiter.filename.startswith("jup")
    assert {"io", "europa", "ganymede", "callisto"} <= set(jupiter.bodies)


def test_group_for_body_routes_correctly() -> None:
    assert group_for_body("earth") is BASE_GROUP
    assert group_for_body("io").name == "jupiter"
    assert group_for_body("titan").name == "saturn"
    assert group_for_body("triton").name == "neptune"
    assert group_for_body("oberon").name == "uranus"


def test_group_for_unknown_body_returns_none() -> None:
    assert group_for_body("xyzzy") is None


def test_group_by_name_unknown_raises() -> None:
    with pytest.raises(KeyError):
        group_by_name("vulcan")


def test_all_groups_have_unique_filenames() -> None:
    seen = [g.filename for g in ALL_GROUPS]
    assert len(seen) == len(set(seen))


def test_satellite_groups_all_have_satellites() -> None:
    for g in SATELLITE_GROUPS:
        assert len(g.bodies) > 0
        for body in g.bodies:
            assert group_for_body(body) is g


# --- manifest I/O -----------------------------------------------------------


def test_manifest_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    expected = {
        "de440s.bsp": {"url": "x", "sha256": "abc", "downloaded_at": "now",
                       "size_bytes": 100, "group": "base"},
    }
    _write_manifest(path, expected)
    actual = _read_manifest(path)
    assert actual == expected


def test_manifest_missing_returns_empty(tmp_path: Path) -> None:
    assert _read_manifest(tmp_path / "no-such.json") == {}


def test_manifest_corrupt_returns_empty(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text("{ not valid json", encoding="utf-8")
    assert _read_manifest(path) == {}


def test_sha256_hashes_a_file(tmp_path: Path) -> None:
    p = tmp_path / "x.bin"
    p.write_bytes(b"hello world")
    # sha256("hello world") = b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9
    assert _sha256(p) == "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
