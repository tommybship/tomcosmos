"""Tests for the ephemeris layer.

Marked @pytest.mark.ephemeris because they require `data/kernels/de440s.bsp`
to be present (the source fixture will skip otherwise).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy import units as u
from astropy.time import Time

from tomcosmos.exceptions import EphemerisOutOfRangeError, UnknownBodyError
from tomcosmos.state.ephemeris import EphemerisSource

pytestmark = pytest.mark.ephemeris

AU_KM = 1.495978707e8


# --- Contract tests --------------------------------------------------------


def test_earth_at_j2026_is_about_1_au_from_sun(ephemeris_source: EphemerisSource) -> None:
    t = Time("2026-04-23T00:00:00", scale="tdb")
    r_earth, _ = ephemeris_source.query("earth", t)
    r_sun, _ = ephemeris_source.query("sun", t)
    r_heliocentric = r_earth - r_sun
    dist_au = float(np.linalg.norm(r_heliocentric)) / AU_KM
    # Earth's orbital eccentricity 0.0167 gives distance in [0.983, 1.017] AU.
    assert 0.98 < dist_au < 1.02


def test_earth_speed_matches_orbital_velocity(
    ephemeris_source: EphemerisSource,
) -> None:
    t = Time("2026-04-23T00:00:00", scale="tdb")
    _, v_earth = ephemeris_source.query("earth", t)
    _, v_sun = ephemeris_source.query("sun", t)
    v_heliocentric_kms = float(np.linalg.norm(v_earth - v_sun))
    # Earth's heliocentric speed is ~29.78 km/s at the mean distance,
    # up to ±0.5 km/s as eccentricity varies it.
    assert 29.0 < v_heliocentric_kms < 30.5


def test_query_by_spice_id_matches_by_name(ephemeris_source: EphemerisSource) -> None:
    t = Time("2026-04-23T00:00:00", scale="tdb")
    r_by_name, v_by_name = ephemeris_source.query("earth", t)
    r_by_id, v_by_id = ephemeris_source.query(399, t)
    assert np.allclose(r_by_name, r_by_id)
    assert np.allclose(v_by_name, v_by_id)


def test_query_returns_shape_3(ephemeris_source: EphemerisSource) -> None:
    t = Time("2026-04-23T00:00:00", scale="tdb")
    r, v = ephemeris_source.query("sun", t)
    assert r.shape == (3,)
    assert v.shape == (3,)
    assert r.dtype == np.float64
    assert v.dtype == np.float64


def test_available_bodies_covers_planet_roster(ephemeris_source: EphemerisSource) -> None:
    bodies = set(ephemeris_source.available_bodies())
    expected = {
        "sun", "mercury", "venus", "earth", "moon",
        "mars", "jupiter", "saturn", "uranus", "neptune",
    }
    assert expected <= bodies


def test_time_range_is_tdb_and_sane(ephemeris_source: EphemerisSource) -> None:
    t_min, t_max = ephemeris_source.time_range()
    assert t_min.scale == "tdb"
    assert t_max.scale == "tdb"
    # DE440s covers roughly 1849-12 to 2149-12; allow some slack.
    assert t_min < Time("1900-01-01", scale="tdb")
    assert t_max > Time("2100-01-01", scale="tdb")


def test_require_covers_accepts_window_inside(
    ephemeris_source: EphemerisSource,
) -> None:
    t = Time("2026-04-23T00:00:00", scale="tdb")
    ephemeris_source.require_covers(t, 1.0 * u.yr)


def test_require_covers_rejects_window_past_end(
    ephemeris_source: EphemerisSource,
) -> None:
    t = Time("2140-01-01T00:00:00", scale="tdb")
    with pytest.raises(EphemerisOutOfRangeError):
        ephemeris_source.require_covers(t, 100.0 * u.yr)


def test_require_covers_rejects_epoch_before_start(
    ephemeris_source: EphemerisSource,
) -> None:
    t = Time("1500-01-01T00:00:00", scale="tdb")
    with pytest.raises(EphemerisOutOfRangeError):
        ephemeris_source.require_covers(t, 1.0 * u.yr)


def test_unknown_body_raises(ephemeris_source: EphemerisSource) -> None:
    t = Time("2026-04-23T00:00:00", scale="tdb")
    with pytest.raises(UnknownBodyError):
        ephemeris_source.query("vulcan", t)


def test_kernel_paths_listed(ephemeris_source: EphemerisSource) -> None:
    paths = ephemeris_source.kernel_paths
    assert len(paths) >= 1
    assert all(p.exists() for p in paths)
    assert any(p.name == "de440s.bsp" for p in paths)


def test_close_is_idempotent(kernel_dir: Path) -> None:
    """`close()` must be safe to call multiple times.

    Builds its own source rather than borrowing the session-scoped
    fixture — closing the shared fixture would cascade into every
    subsequent test."""
    src = EphemerisSource(directory=kernel_dir)
    src.close()
    src.close()  # must not raise


def test_kernel_paths_base_first(ephemeris_source: EphemerisSource) -> None:
    """`kernel_paths` exposes the base (DE44x) kernel as paths[0]; this
    is load-bearing for diagnostics and run-metadata reproducibility."""
    paths = ephemeris_source.kernel_paths
    assert paths[0].name == "de440s.bsp"


def test_galilean_query_without_kernel_says_how_to_install(
    ephemeris_source: EphemerisSource, kernel_dir: Path,
) -> None:
    """When jup365.bsp isn't present, asking for Io should fail with a
    clear instruction to fetch the right kernel — not a generic error.

    Skipped when jup365.bsp IS loaded; the missing-kernel error path
    is what we're testing here, not the success path."""
    if (kernel_dir / "jup365.bsp").exists():
        pytest.skip("jup365.bsp present; can't exercise missing-kernel error")
    t = Time("2026-04-23T00:00:00", scale="tdb")
    with pytest.raises(UnknownBodyError, match=r"jup.*\.bsp|fetch-kernels --include jupiter"):
        ephemeris_source.query("io", t)


def test_titan_query_without_kernel_says_how_to_install(
    ephemeris_source: EphemerisSource, kernel_dir: Path,
) -> None:
    if (kernel_dir / "sat441.bsp").exists():
        pytest.skip("sat441.bsp present; can't exercise missing-kernel error")
    t = Time("2026-04-23T00:00:00", scale="tdb")
    with pytest.raises(UnknownBodyError, match=r"sat.*\.bsp|fetch-kernels --include saturn"):
        ephemeris_source.query("titan", t)


# --- Parented-body positions are SSB-relative, not double-counted ----------

# Each Galilean's mean orbital semi-major axis around Jupiter, in AU. The test
# accepts up to 2× this as a generous bound: we're catching the "moon at 5+ AU
# from primary" double-count bug, not asserting per-body precision.
_GALILEAN_RADIUS_AU = {
    "io":       0.00282,  #  421,800 km
    "europa":   0.00449,  #  671,100 km
    "ganymede": 0.00715,  # 1,070,000 km
    "callisto": 0.01259,  # 1,883,000 km
}


@pytest.mark.parametrize("moon", list(_GALILEAN_RADIUS_AU))
def test_galilean_is_close_to_jupiter(
    ephemeris_source: EphemerisSource, kernel_dir: Path, moon: str,
) -> None:
    """A satellite-kernel body must end up near its primary, not at 2× the
    primary's SSB distance.

    Regression for the chained-add bug in pre-2026-04 EphemerisSource: the
    library already returns SSB-relative positions for parented bodies, so
    adding the parent's SSB position on top placed every Galilean at twice
    Jupiter's distance. Bound: 2× the moon's real orbital radius."""
    if not (kernel_dir / "jup365.bsp").exists():
        pytest.skip("jup365.bsp not present")
    t = Time("2026-04-23T00:00:00", scale="tdb")
    r_moon, _ = ephemeris_source.query(moon, t)
    r_jup, _ = ephemeris_source.query("jupiter", t)
    dist_au = float(np.linalg.norm(r_moon - r_jup)) / AU_KM
    bound = 2.0 * _GALILEAN_RADIUS_AU[moon]
    assert dist_au < bound, (
        f"{moon} is {dist_au:.4f} AU from jupiter, expected ~{_GALILEAN_RADIUS_AU[moon]:.4f} AU; "
        f"if dist ≈ jupiter's SSB distance the chained-add bug has returned"
    )
