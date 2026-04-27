"""Measure Mercury's perihelion precession from two 100-yr runs.

Reads two parquets — typically `sun-planets-100yr.yaml` (Newtonian) and
`sun-planets-100yr-gr.yaml` (Newtonian + REBOUNDx 1PN GR) — and prints
the perihelion-longitude drift rate for Mercury in each, then the
difference. The expected pattern:

    Newtonian planetary perturbations:  ~530"/century
    + 1PN GR around the Sun:            +43"/century
    -----------------------------------------------
    Total Newton+GR:                    ~573"/century
    Difference (GR contribution):       ~43"/century

The 43"/century is the historic Le Verrier residual that GR resolved.

Method: at each sample, compute Mercury's heliocentric orbit elements
from (r, v) → eccentricity vector → atan2(e_y, e_x) in the ICRF
ecliptic plane → unwrap → linear fit ϖ vs. t → drift rate.

Usage:
    python scripts/mercury_precession.py \\
        runs/sun-planets-100yr__*.parquet \\
        runs/sun-planets-100yr-gr__*.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from tomcosmos.io.history import StateHistory

# JPL Astrodynamic Parameters: Sun's GM in km^3/s^2.
# (BODY_CONSTANTS uses mass*G — same to ~1e-7, fine for precession rates.)
_MU_SUN_KM3_S2: float = 1.32712440018e11

_SECONDS_PER_CENTURY: float = 100.0 * 365.25 * 86400.0
_RAD_TO_ARCSEC: float = (180.0 / np.pi) * 3600.0


def _heliocentric_state(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-sample (t_sec, r_helio_km, v_helio_kms) for Mercury."""
    sun = df[df["body"] == "sun"].sort_values("sample_idx")
    mer = df[df["body"] == "mercury"].sort_values("sample_idx")
    t_sec = mer["t_tdb"].to_numpy(dtype=np.float64)
    r = mer[["x", "y", "z"]].to_numpy(dtype=np.float64) - sun[["x", "y", "z"]].to_numpy(dtype=np.float64)
    v = mer[["vx", "vy", "vz"]].to_numpy(dtype=np.float64) - sun[["vx", "vy", "vz"]].to_numpy(dtype=np.float64)
    return t_sec, r, v


def _perihelion_longitude_rad(r: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Per-sample perihelion-longitude angle in the ICRF X-Y plane.

    e_vec = (v × h)/μ - r̂  points from focus to perihelion; its
    azimuth in ICRF X-Y is what precesses. atan2 unwrapped to a
    continuous radians sequence.
    """
    h = np.cross(r, v)  # specific angular momentum, km^2/s
    r_norm = np.linalg.norm(r, axis=1, keepdims=True)
    e_vec = np.cross(v, h) / _MU_SUN_KM3_S2 - r / r_norm
    omega = np.arctan2(e_vec[:, 1], e_vec[:, 0])  # ICRF Y/X
    return np.unwrap(omega)


def _drift_arcsec_per_century(t_sec: np.ndarray, omega_rad: np.ndarray) -> float:
    """Linear fit ϖ(t) and return slope as arcsec / century."""
    slope_rad_per_sec, _intercept = np.polyfit(t_sec, omega_rad, 1)
    return float(slope_rad_per_sec * _SECONDS_PER_CENTURY * _RAD_TO_ARCSEC)


def main() -> None:
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)

    newton_path = Path(sys.argv[1])
    gr_path = Path(sys.argv[2])

    newton = StateHistory.from_parquet(newton_path)
    gr = StateHistory.from_parquet(gr_path)

    if newton.scenario.epoch.tdb.jd != gr.scenario.epoch.tdb.jd:
        raise SystemExit(
            f"epoch mismatch: {newton.scenario.epoch} vs {gr.scenario.epoch}"
        )

    t_n, r_n, v_n = _heliocentric_state(newton.df)
    t_g, r_g, v_g = _heliocentric_state(gr.df)

    omega_n = _perihelion_longitude_rad(r_n, v_n)
    omega_g = _perihelion_longitude_rad(r_g, v_g)

    rate_n = _drift_arcsec_per_century(t_n, omega_n)
    rate_g = _drift_arcsec_per_century(t_g, omega_g)
    delta = rate_g - rate_n

    print(f"Mercury perihelion precession over {t_n[-1] / _SECONDS_PER_CENTURY * 100:.1f} yr:")
    print(f"  Newtonian only      : {rate_n:>9.1f} arcsec/century")
    print(f"  Newtonian + 1PN GR  : {rate_g:>9.1f} arcsec/century")
    print(f"  GR contribution     : {delta:>9.1f} arcsec/century  (textbook: 43.0)")


if __name__ == "__main__":
    main()
