"""Coordinate-frame conversions.

The internal frame is ICRF (International Celestial Reference Frame),
origin at the solar system barycenter (SSB). Everything the integrator
touches is in that frame. This module provides pure rotations to and
from the ecliptic J2000 frame; barycentric↔heliocentric translations
need the Sun's state vector, so they live in `ic.py` where we already
have an ephemeris handy.

All functions operate on plain NumPy arrays (km, km/s). No astropy
Quantity wrapping inside the physics core — units live in names.

Obliquity at J2000 epoch: ε = 23°26'21.448" = 23.4392911° (IAU 1976).
This is the angle that tilts the ecliptic relative to the equatorial
(ICRF) frame. Modern IAU 2006/2010 obliquity differs in the 4th decimal
place; ignored here (see PLAN.md > Non-goals — learning-grade).
"""
from __future__ import annotations

import numpy as np

OBLIQUITY_RAD: float = np.deg2rad(23.4392911)

_COS_EPS = float(np.cos(OBLIQUITY_RAD))
_SIN_EPS = float(np.sin(OBLIQUITY_RAD))

# Rotation that takes ecliptic J2000 coordinates to equatorial (ICRF) J2000.
# x-axis is shared (vernal equinox); the ecliptic is tilted about x by ε,
# so the rotation about x by +ε lifts ecliptic Y,Z into equatorial Y,Z.
_R_ECL_TO_ICRF: np.ndarray = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, _COS_EPS, -_SIN_EPS],
        [0.0, _SIN_EPS, _COS_EPS],
    ],
    dtype=np.float64,
)
_R_ICRF_TO_ECL: np.ndarray = _R_ECL_TO_ICRF.T


def ecliptic_to_icrf(vec: np.ndarray) -> np.ndarray:
    """Rotate a 3-vector (or N,3 array) from ecliptic J2000 to ICRF/equatorial J2000."""
    v = np.asarray(vec, dtype=np.float64)
    if v.shape[-1] != 3:
        raise ValueError(f"last axis must be length 3, got shape {v.shape}")
    return v @ _R_ECL_TO_ICRF.T


def icrf_to_ecliptic(vec: np.ndarray) -> np.ndarray:
    """Rotate a 3-vector (or N,3 array) from ICRF/equatorial J2000 to ecliptic J2000."""
    v = np.asarray(vec, dtype=np.float64)
    if v.shape[-1] != 3:
        raise ValueError(f"last axis must be length 3, got shape {v.shape}")
    return v @ _R_ICRF_TO_ECL.T
