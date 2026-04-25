"""Tests for the ephemeris layer.

Marked @pytest.mark.ephemeris because they require `data/kernels/de440s.bsp`
to be present (the `skyfield_source` fixture will skip otherwise).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy import units as u
from astropy.time import Time

from tomcosmos.exceptions import EphemerisOutOfRangeError, UnknownBodyError
from tomcosmos.state.ephemeris import SkyfieldSource

pytestmark = pytest.mark.ephemeris

AU_KM = 1.495978707e8


def test_earth_at_j2026_is_about_1_au_from_sun(skyfield_source: SkyfieldSource) -> None:
    t = Time("2026-04-23T00:00:00", scale="tdb")
    r_earth, _ = skyfield_source.query("earth", t)
    r_sun, _ = skyfield_source.query("sun", t)
    r_heliocentric = r_earth - r_sun
    dist_au = float(np.linalg.norm(r_heliocentric)) / AU_KM
    # Earth's orbital eccentricity 0.0167 gives distance in [0.983, 1.017] AU.
    assert 0.98 < dist_au < 1.02


def test_earth_speed_matches_orbital_velocity(
    skyfield_source: SkyfieldSource,
) -> None:
    t = Time("2026-04-23T00:00:00", scale="tdb")
    _, v_earth = skyfield_source.query("earth", t)
    _, v_sun = skyfield_source.query("sun", t)
    v_heliocentric_kms = float(np.linalg.norm(v_earth - v_sun))
    # Earth's heliocentric speed is ~29.78 km/s at the mean distance,
    # up to ±0.5 km/s as eccentricity varies it.
    assert 29.0 < v_heliocentric_kms < 30.5


def test_query_by_spice_id_matches_by_name(skyfield_source: SkyfieldSource) -> None:
    t = Time("2026-04-23T00:00:00", scale="tdb")
    r_by_name, v_by_name = skyfield_source.query("earth", t)
    r_by_id, v_by_id = skyfield_source.query(399, t)
    assert np.allclose(r_by_name, r_by_id)
    assert np.allclose(v_by_name, v_by_id)


def test_query_returns_shape_3(skyfield_source: SkyfieldSource) -> None:
    t = Time("2026-04-23T00:00:00", scale="tdb")
    r, v = skyfield_source.query("sun", t)
    assert r.shape == (3,)
    assert v.shape == (3,)
    assert r.dtype == np.float64
    assert v.dtype == np.float64


def test_available_bodies_covers_m1_roster(skyfield_source: SkyfieldSource) -> None:
    bodies = set(skyfield_source.available_bodies())
    expected = {
        "sun", "mercury", "venus", "earth", "moon",
        "mars", "jupiter", "saturn", "uranus", "neptune",
    }
    assert expected <= bodies


def test_time_range_is_tdb_and_sane(skyfield_source: SkyfieldSource) -> None:
    t_min, t_max = skyfield_source.time_range()
    assert t_min.scale == "tdb"
    assert t_max.scale == "tdb"
    # DE440s covers roughly 1849-12 to 2149-12; allow some slack.
    assert t_min < Time("1900-01-01", scale="tdb")
    assert t_max > Time("2100-01-01", scale="tdb")


def test_require_covers_accepts_window_inside(skyfield_source: SkyfieldSource) -> None:
    t = Time("2026-04-23T00:00:00", scale="tdb")
    skyfield_source.require_covers(t, 1.0 * u.yr)


def test_require_covers_rejects_window_past_end(
    skyfield_source: SkyfieldSource,
) -> None:
    t = Time("2140-01-01T00:00:00", scale="tdb")
    with pytest.raises(EphemerisOutOfRangeError):
        skyfield_source.require_covers(t, 100.0 * u.yr)


def test_require_covers_rejects_epoch_before_start(
    skyfield_source: SkyfieldSource,
) -> None:
    t = Time("1500-01-01T00:00:00", scale="tdb")
    with pytest.raises(EphemerisOutOfRangeError):
        skyfield_source.require_covers(t, 1.0 * u.yr)


def test_unknown_body_raises(skyfield_source: SkyfieldSource) -> None:
    t = Time("2026-04-23T00:00:00", scale="tdb")
    with pytest.raises(UnknownBodyError):
        skyfield_source.query("vulcan", t)


# --- M2: multi-kernel routing -----------------------------------------------


def test_galilean_query_without_kernel_says_how_to_install(
    skyfield_source: SkyfieldSource, kernel_dir: Path,
) -> None:
    """When jup365.bsp isn't present, asking for Io should fail with a
    clear instruction to fetch the right kernel — not a generic error.

    Skipped when jup365.bsp IS loaded; the missing-kernel error path
    is what we're testing here, not the success path (covered elsewhere)."""
    if (kernel_dir / "jup365.bsp").exists():
        pytest.skip("jup365.bsp present; can't exercise missing-kernel error")
    t = Time("2026-04-23T00:00:00", scale="tdb")
    with pytest.raises(UnknownBodyError, match=r"jup.*\.bsp|fetch-kernels --include jupiter"):
        skyfield_source.query("io", t)


def test_titan_query_without_kernel_says_how_to_install(
    skyfield_source: SkyfieldSource, kernel_dir: Path,
) -> None:
    if (kernel_dir / "sat441.bsp").exists():
        pytest.skip("sat441.bsp present; can't exercise missing-kernel error")
    t = Time("2026-04-23T00:00:00", scale="tdb")
    with pytest.raises(UnknownBodyError, match=r"sat.*\.bsp|fetch-kernels --include saturn"):
        skyfield_source.query("titan", t)


def test_kernel_paths_lists_loaded_kernels(skyfield_source: SkyfieldSource) -> None:
    paths = skyfield_source.kernel_paths
    assert len(paths) >= 1
    # Base kernel always first
    assert paths[0].name == "de440s.bsp"
