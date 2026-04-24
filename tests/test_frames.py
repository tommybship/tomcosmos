import numpy as np
import pytest

from tomcosmos.state.frames import (
    OBLIQUITY_RAD,
    ecliptic_to_icrf,
    icrf_to_ecliptic,
)


def test_obliquity_about_j2000() -> None:
    assert np.isclose(np.rad2deg(OBLIQUITY_RAD), 23.4392911, atol=1e-7)


def test_roundtrip_single_vector() -> None:
    rng = np.random.default_rng(0)
    v = rng.standard_normal(3) * 1e8
    v2 = ecliptic_to_icrf(icrf_to_ecliptic(v))
    assert np.allclose(v, v2, atol=1e-9)


def test_roundtrip_array() -> None:
    rng = np.random.default_rng(1)
    v = rng.standard_normal((100, 3)) * 1e8
    v2 = ecliptic_to_icrf(icrf_to_ecliptic(v))
    assert np.allclose(v, v2, atol=1e-9)


def test_x_axis_is_fixed() -> None:
    x = np.array([1.0, 0.0, 0.0])
    assert np.allclose(ecliptic_to_icrf(x), x)
    assert np.allclose(icrf_to_ecliptic(x), x)


def test_ecliptic_y_lifts_toward_equatorial_z() -> None:
    y_ecl = np.array([0.0, 1.0, 0.0])
    y_icrf = ecliptic_to_icrf(y_ecl)
    # Ecliptic +Y rotates about +X by +ε, landing at (0, cos ε, sin ε).
    assert np.isclose(y_icrf[0], 0.0)
    assert np.isclose(y_icrf[1], np.cos(OBLIQUITY_RAD))
    assert np.isclose(y_icrf[2], np.sin(OBLIQUITY_RAD))


def test_ecliptic_z_tilts_away_from_icrf_z() -> None:
    z_ecl = np.array([0.0, 0.0, 1.0])
    z_icrf = ecliptic_to_icrf(z_ecl)
    # Ecliptic +Z → (0, -sin ε, cos ε).
    assert np.isclose(z_icrf[0], 0.0)
    assert np.isclose(z_icrf[1], -np.sin(OBLIQUITY_RAD))
    assert np.isclose(z_icrf[2], np.cos(OBLIQUITY_RAD))


def test_rejects_wrong_last_axis() -> None:
    with pytest.raises(ValueError, match="last axis"):
        ecliptic_to_icrf(np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="last axis"):
        icrf_to_ecliptic(np.array([[1.0, 2.0], [3.0, 4.0]]))


def test_preserves_magnitude() -> None:
    rng = np.random.default_rng(2)
    v = rng.standard_normal(3) * 1e7
    mag_in = np.linalg.norm(v)
    mag_out = np.linalg.norm(ecliptic_to_icrf(v))
    assert np.isclose(mag_in, mag_out, rtol=1e-12)
