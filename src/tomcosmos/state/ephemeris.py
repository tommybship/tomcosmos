"""Ephemeris sources — where body state vectors come from.

Single ABC with M1's skyfield backend. M2 adds `SpiceSource` (spiceypy)
when satellite kernels enter the picture; the ABC contract is the same
so scenario code doesn't care which backend is loaded.

Queries return ICRF barycentric (r_km, v_kms) as shape-(3,) arrays.
Outer planets (Jupiter-Neptune) resolve to the *system barycenter*
rather than the planet center — for M1 (planets only, no moons) this
is the standard pragmatic choice and the offset is < ~500 km for
Neptune, much less for the rest.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.time import Time
from astropy.units import Quantity

from tomcosmos.config import kernel_dir as default_kernel_dir
from tomcosmos.constants import BodyConstant, resolve_body_constant
from tomcosmos.exceptions import EphemerisOutOfRangeError, UnknownBodyError


class EphemerisSource(ABC):
    """Abstract interface for a source of ICRF-barycentric body states.

    Backends:
      - `SkyfieldSource` — M1+, reads DE44x SPK via skyfield.
      - `SpiceSource` — M2+, reads same SPKs via spiceypy; adds satellite kernels.
    """

    @abstractmethod
    def query(self, body: str | int, epoch: Time) -> tuple[np.ndarray, np.ndarray]:
        """Return (r_km, v_kms) in ICRF barycentric at `epoch`. Shapes: (3,)."""

    @abstractmethod
    def available_bodies(self) -> tuple[str, ...]:
        """Canonical lowercase names of bodies this source can resolve."""

    @abstractmethod
    def time_range(self) -> tuple[Time, Time]:
        """(t_min, t_max) TDB bounds this source can query over.

        For multi-segment kernels, this is the intersection across all
        requested bodies (safest: the tightest common window).
        """

    def require_covers(self, epoch: Time, duration: Quantity) -> None:
        """Raise `EphemerisOutOfRangeError` if `epoch + duration` escapes coverage."""
        t_min, t_max = self.time_range()
        t_end = epoch + duration.to(u.s)
        if epoch < t_min or t_end > t_max:
            raise EphemerisOutOfRangeError(
                f"scenario window [{epoch.isot}, {t_end.isot}] "
                f"outside ephemeris coverage [{t_min.isot}, {t_max.isot}]"
            )


# de440s.bsp contents (as of 2024): planet centers 199/299/399 for Mercury,
# Venus, Earth; 301 Moon; 10 Sun; and system barycenters 1..9 for everyone
# else (including 4 Mars — Mars center 499 is not in the small kernel).
# Using the barycenter for Mars and the outer planets is correct anyway
# when we pair it with the planet-only mass from constants.py:
#   - Inner planets: center ≈ barycenter (no significant moons).
#   - Mars: moons are ~1e-8 of Mars's mass, so offset is negligible.
#   - Outer planets: moon systems are 1e-4..1e-3 of planet mass; using
#     barycenter position with planet-only mass leaves a small residual,
#     well inside the learning-grade envelope (see PLAN.md > Non-goals).
_SKYFIELD_KEY: dict[str, str] = {
    "sun": "sun",
    "mercury": "mercury",
    "venus": "venus",
    "earth": "earth",
    "moon": "moon",
    "mars": "mars barycenter",
    "jupiter": "jupiter barycenter",
    "saturn": "saturn barycenter",
    "uranus": "uranus barycenter",
    "neptune": "neptune barycenter",
}


class SkyfieldSource(EphemerisSource):
    """DE44x ephemeris via skyfield; M1 default."""

    def __init__(
        self,
        kernel_filename: str = "de440s.bsp",
        directory: str | Path | None = None,
    ) -> None:
        from skyfield.api import Loader

        d = Path(directory) if directory is not None else default_kernel_dir()
        d.mkdir(parents=True, exist_ok=True)
        self._directory = d
        self._kernel_filename = kernel_filename
        self._loader = Loader(str(d))
        self._kernel = self._loader(kernel_filename)
        self._ts = self._loader.timescale()

    @property
    def kernel_path(self) -> Path:
        return self._directory / self._kernel_filename

    def _resolve_key(self, body: str | int) -> str:
        const: BodyConstant = resolve_body_constant(body)
        key = _SKYFIELD_KEY.get(const.name)
        if key is None:
            raise UnknownBodyError(
                f"body {const.name!r} has no skyfield mapping (known: {sorted(_SKYFIELD_KEY)})"
            )
        return key

    def query(self, body: str | int, epoch: Time) -> tuple[np.ndarray, np.ndarray]:
        key = self._resolve_key(body)
        # skyfield's tdb_jd takes a single float; astropy Time preserves precision
        # via two-part jd but we pass the combined value — sub-ms precision loss is
        # well inside our learning-grade envelope.
        t = self._ts.tdb_jd(float(epoch.tdb.jd))
        pos = self._kernel[key].at(t)
        r_km = np.asarray(pos.position.km, dtype=np.float64)
        v_kms = np.asarray(pos.velocity.km_per_s, dtype=np.float64)
        return r_km, v_kms

    def available_bodies(self) -> tuple[str, ...]:
        # All bodies in our SPICE_KEY map that the loaded kernel actually contains.
        available: list[str] = []
        for name, key in _SKYFIELD_KEY.items():
            try:
                _ = self._kernel[key]
            except (KeyError, ValueError):
                continue
            available.append(name)
        return tuple(available)

    def time_range(self) -> tuple[Time, Time]:
        # Take the intersection of all SPK segment windows. start_jd/end_jd are TDB.
        segments = self._kernel.spk.segments
        if not segments:
            raise RuntimeError(
                f"no SPK segments found in {self.kernel_path}; is the file a valid SPK?"
            )
        start_jd = max(seg.start_jd for seg in segments)
        end_jd = min(seg.end_jd for seg in segments)
        return (
            Time(start_jd, format="jd", scale="tdb"),
            Time(end_jd, format="jd", scale="tdb"),
        )
