"""Ephemeris sources — where body state vectors come from.

Single ABC with a multi-kernel skyfield backend. M2+ adds satellite
kernels (jup365, sat441, ...) which are downloaded opt-in via
`fetch-kernels --include <group>`. The backend transparently chains
across kernels: `query("io", epoch)` resolves Io's position relative
to Jupiter system barycenter from `jup365.bsp`, then adds the Jupiter
barycenter → SSB vector from `de440s.bsp`, returning Io in ICRF
barycentric.

Queries return ICRF barycentric (r_km, v_kms) as shape-(3,) arrays.
For outer planets in M1 default scenarios (Jupiter–Neptune) we
resolve to the *system barycenter* rather than the planet center
when the satellite kernel isn't loaded. With it loaded, the planet
center is also reachable; canonical body names (`jupiter`, etc.)
still map to the barycenter for backward compatibility with M1.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, NoReturn

import numpy as np
from astropy import units as u
from astropy.time import Time
from astropy.units import Quantity

from tomcosmos.config import kernel_dir as default_kernel_dir
from tomcosmos.constants import BodyConstant, resolve_body_constant
from tomcosmos.exceptions import EphemerisOutOfRangeError, UnknownBodyError
from tomcosmos.kernels import group_for_body


class EphemerisSource(ABC):
    """Abstract interface for a source of ICRF-barycentric body states.

    Backends:
      - `SkyfieldSource` — multi-kernel; loads every .bsp in the
        kernel directory.
      - `SpiceSource` — future, via spiceypy. Same contract.
    """

    @abstractmethod
    def query(self, body: str | int, epoch: Time) -> tuple[np.ndarray, np.ndarray]:
        """Return (r_km, v_kms) in ICRF barycentric at `epoch`. Shapes: (3,)."""

    @abstractmethod
    def available_bodies(self) -> tuple[str, ...]:
        """Canonical lowercase names of bodies this source can resolve."""

    @abstractmethod
    def time_range(self) -> tuple[Time, Time]:
        """(t_min, t_max) TDB bounds. Intersection across all loaded kernels."""

    def require_covers(self, epoch: Time, duration: Quantity) -> None:
        """Raise `EphemerisOutOfRangeError` if `epoch + duration` escapes coverage."""
        t_min, t_max = self.time_range()
        t_end = epoch + duration.to(u.s)
        if epoch < t_min or t_end > t_max:
            raise EphemerisOutOfRangeError(
                f"scenario window [{epoch.isot}, {t_end.isot}] "
                f"outside ephemeris coverage [{t_min.isot}, {t_max.isot}]"
            )


# Canonical body name → (skyfield-key, parent-canonical-name).
# parent=None means the kernel directly provides this body relative to SSB.
# parent="jupiter" means we look up <body> in its own kernel (Jupiter system
# barycenter relative — that's how NAIF satellite kernels are structured)
# and add the Jupiter-barycenter→SSB vector from de440s.
_SKYFIELD_RESOLVERS: dict[str, tuple[str, str | None]] = {
    # In-DE440s bodies — direct lookup.
    "sun":       ("sun", None),
    "mercury":   ("mercury", None),
    "venus":     ("venus", None),
    "earth":     ("earth", None),
    "moon":      ("moon", None),
    "mars":      ("mars barycenter", None),
    "jupiter":   ("jupiter barycenter", None),
    "saturn":    ("saturn barycenter", None),
    "uranus":    ("uranus barycenter", None),
    "neptune":   ("neptune barycenter", None),
    # Galilean moons — kernel skyfield key is just the moon's name; parent is
    # Jupiter barycenter (5) which we get from de440s and add.
    "io":       ("io", "jupiter"),
    "europa":   ("europa", "jupiter"),
    "ganymede": ("ganymede", "jupiter"),
    "callisto": ("callisto", "jupiter"),
    # Saturnian moons.
    "mimas":     ("mimas", "saturn"),
    "enceladus": ("enceladus", "saturn"),
    "tethys":    ("tethys", "saturn"),
    "dione":     ("dione", "saturn"),
    "rhea":      ("rhea", "saturn"),
    "titan":     ("titan", "saturn"),
    "iapetus":   ("iapetus", "saturn"),
    # Uranian moons.
    "titania": ("titania", "uranus"),
    "oberon":  ("oberon", "uranus"),
    # Neptunian moons.
    "triton": ("triton", "neptune"),
}


class SkyfieldSource(EphemerisSource):
    """Multi-kernel ephemeris via skyfield. Loads every `*.bsp` in the
    kernel directory and routes queries to whichever file provides the
    requested body, chaining position vectors across kernels when needed.
    """

    def __init__(
        self,
        kernel_filename: str = "de440s.bsp",
        directory: str | Path | None = None,
    ) -> None:
        from skyfield.api import Loader

        d = Path(directory) if directory is not None else default_kernel_dir()
        d.mkdir(parents=True, exist_ok=True)
        self._directory = d
        # Backward-compat: kernel_filename names the *base* (DE44x) kernel.
        # All other .bsp files in the directory are loaded too if present.
        self._kernel_filename = kernel_filename
        self._loader = Loader(str(d))
        self._kernels: dict[str, Any] = {}  # filename -> skyfield SpiceKernel

        # Load the base kernel first (required).
        self._kernels[kernel_filename] = self._loader(kernel_filename)

        # Discover and load every other .bsp in the directory.
        for path in sorted(d.glob("*.bsp")):
            if path.name == kernel_filename:
                continue
            self._kernels[path.name] = self._loader(path.name)

        self._ts = self._loader.timescale()

    @property
    def kernel_path(self) -> Path:
        """Path to the base kernel — kept for back-compat with diagnostics."""
        return self._directory / self._kernel_filename

    @property
    def kernel_paths(self) -> tuple[Path, ...]:
        """All loaded kernel paths, base first."""
        return tuple(self._directory / name for name in self._kernels)

    # ------------------------------------------------------------------

    def _resolver_for(self, body: str | int) -> tuple[str, str | None]:
        const: BodyConstant = resolve_body_constant(body)
        resolver = _SKYFIELD_RESOLVERS.get(const.name)
        if resolver is None:
            raise UnknownBodyError(
                f"body {const.name!r} has no skyfield resolver "
                f"(known: {sorted(_SKYFIELD_RESOLVERS)})"
            )
        return resolver

    def _kernel_with(self, skyfield_key: str) -> Any | None:
        """Return the first loaded kernel that knows about `skyfield_key`."""
        for kernel in self._kernels.values():
            try:
                _ = kernel[skyfield_key]
            except (KeyError, ValueError):
                continue
            return kernel
        return None

    # ------------------------------------------------------------------

    def query(self, body: str | int, epoch: Time) -> tuple[np.ndarray, np.ndarray]:
        target_key, parent_canonical = self._resolver_for(body)
        t = self._ts.tdb_jd(float(epoch.tdb.jd))

        # Find the kernel that has the target.
        target_kernel = self._kernel_with(target_key)
        if target_kernel is None:
            self._raise_missing_kernel(body, target_key, parent_canonical)

        target_pos = target_kernel[target_key].at(t)
        r = np.asarray(target_pos.position.km, dtype=np.float64)
        v = np.asarray(target_pos.velocity.km_per_s, dtype=np.float64)

        if parent_canonical is None:
            # Direct: target is given relative to SSB by the kernel.
            return r, v

        # Chained: add the parent's SSB-relative state from de440s.
        parent_target, parent_parent = _SKYFIELD_RESOLVERS[parent_canonical]
        parent_kernel = self._kernel_with(parent_target)
        if parent_kernel is None:
            raise UnknownBodyError(
                f"body {body!r} resolves through parent {parent_canonical!r} "
                f"which isn't loaded in any kernel"
            )
        # parent_parent should always be None for our currently-mapped parents
        # (they all live in de440s relative to SSB). If a future body has a
        # 3-deep chain, this recursion adds another leg — keep it simple here.
        parent_pos = parent_kernel[parent_target].at(t)
        r += np.asarray(parent_pos.position.km, dtype=np.float64)
        v += np.asarray(parent_pos.velocity.km_per_s, dtype=np.float64)
        return r, v

    def available_bodies(self) -> tuple[str, ...]:
        out: list[str] = []
        for canonical, (target_key, parent_canonical) in _SKYFIELD_RESOLVERS.items():
            if self._kernel_with(target_key) is None:
                continue
            if parent_canonical is not None:
                parent_target = _SKYFIELD_RESOLVERS[parent_canonical][0]
                if self._kernel_with(parent_target) is None:
                    continue  # have the moon kernel but not the planet's
            out.append(canonical)
        return tuple(out)

    def time_range(self) -> tuple[Time, Time]:
        # Per-kernel span = (min start_jd, max end_jd) across that kernel's
        # segments. Multi-piece kernels (e.g., jup365 has segments that join
        # at boundary epochs) are unioned within a kernel — we trust skyfield
        # to pick the right segment at query time. The cross-kernel result
        # is then the *intersection* of per-kernel spans, since each body
        # we'd query lives in exactly one kernel and we need every queried
        # body to be covered.
        per_kernel_spans: list[tuple[float, float]] = []
        for kernel in self._kernels.values():
            segs = kernel.spk.segments
            if not segs:
                continue
            per_kernel_spans.append(
                (min(s.start_jd for s in segs), max(s.end_jd for s in segs))
            )
        if not per_kernel_spans:
            raise RuntimeError(
                f"no SPK segments found across {len(self._kernels)} loaded kernels"
            )
        start_jd = max(span[0] for span in per_kernel_spans)
        end_jd = min(span[1] for span in per_kernel_spans)
        return (
            Time(start_jd, format="jd", scale="tdb"),
            Time(end_jd, format="jd", scale="tdb"),
        )

    # ------------------------------------------------------------------

    def _raise_missing_kernel(
        self, body: str | int, target_key: str, parent_canonical: str | None
    ) -> NoReturn:
        """Raise UnknownBodyError naming the specific kernel group to install."""
        const = resolve_body_constant(body)
        group = group_for_body(const.name)
        if group is None:
            raise UnknownBodyError(
                f"body {const.name!r} (skyfield key {target_key!r}) isn't in any "
                "loaded kernel and isn't in our kernel registry. Add to "
                "tomcosmos.kernels and re-run fetch-kernels."
            )
        raise UnknownBodyError(
            f"body {const.name!r} requires kernel '{group.filename}' "
            f"(~{group.approx_size_mb:.0f} MB). Run:\n"
            f"  tomcosmos fetch-kernels --include {group.name}\n"
            f"to download it. Currently loaded: "
            f"{[k for k in self._kernels]}"
        )
