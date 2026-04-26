"""Ephemeris source — where body state vectors come from at t₀.

Single backend: skyfield. Reads NAIF SPK kernels (DE440s for the planets /
Sun / Moon, plus optional satellite kernels jup365 / sat441 / nep097 /
plu060 / mar099 for moons of the giant planets and Mars) to provide
ICRF barycentric `(r_km, v_kms)` for any body covered by a loaded
kernel.

This is **only** used for initial-condition resolution at scenario
epoch in Mode B (vanilla REBOUND + REBOUNDx). Mode A (ASSIST) reads
its own DE440 / sb441-n16 directly inside the force loop and does not
go through this module.

Why skyfield and not something else:
  - Pure Python; builds on Windows out of the box.
  - Reads arbitrary NAIF SPK files, including the satellite kernels
    that ASSIST's hardcoded body table does not cover (Galilean moons,
    Titan, Triton, Pluto-system moons, etc.).
  - Cross-checked against JPL Horizons in `tests/test_ephemeris.py`.

**On the chained-kernel question**: NAIF satellite kernels (e.g. jup365)
store moons relative to their planet's system barycenter, but every
modern toolkit that walks an SPK chain (skyfield's segment graph) returns
SSB-relative coordinates when both the moon kernel and the corresponding
base kernel (de440s) are loaded. Earlier versions of this file added the
parent's offset on top of the already-SSB-relative moon position, double-
counting. Don't.
"""
from __future__ import annotations

from pathlib import Path
from types import TracebackType
from typing import Any, NoReturn

import numpy as np
from astropy import units as u
from astropy.time import Time
from astropy.units import Quantity

from tomcosmos.config import kernel_dir as default_kernel_dir
from tomcosmos.constants import BodyConstant, resolve_body_constant
from tomcosmos.exceptions import EphemerisOutOfRangeError, UnknownBodyError
from tomcosmos.kernels import group_for_body

# Canonical body name → (skyfield-key, parent-canonical-name).
# parent=None means the body is queried directly from the base kernel.
# parent="jupiter" means the body lives in a satellite kernel (e.g. jup365)
# and skyfield needs the parent's kernel loaded too so it can chain its
# internal lookup back to SSB. We do not add the parent's position
# ourselves — skyfield already does that. The parent column drives
# availability and error messages, not the math.
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


class EphemerisSource:
    """Multi-kernel ephemeris via skyfield. Loads every `*.bsp` in the
    kernel directory and routes queries to whichever file provides the
    requested body, chaining position vectors across kernels when needed.

    Single concrete class — there used to be an ABC with two backends
    (skyfield and spiceypy) but they did identical jobs and the parallel
    rails earned no keep. See PLAN.md > "Mode A vs Mode B" for the
    division of labor between this module and ASSIST.
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
        # `kernel_filename` names the *base* (DE44x) kernel; all other .bsp
        # files in the directory are loaded too if present.
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

    # ------------------------------------------------------------------
    # Context manager + lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:  # noqa: B027 — skyfield has no per-source teardown
        """No-op. Skyfield keeps no global mutable state we need to release."""

    def __enter__(self) -> EphemerisSource:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    @property
    def kernel_paths(self) -> tuple[Path, ...]:
        """All loaded kernel paths, base first.

        Used by run-metadata diagnostics to SHA256 the inputs for
        reproducibility. Order is stable but otherwise unspecified.
        """
        return tuple(self._directory / name for name in self._kernels)

    def require_covers(self, epoch: Time, duration: Quantity) -> None:
        """Raise `EphemerisOutOfRangeError` if `epoch + duration` escapes coverage."""
        t_min, t_max = self.time_range()
        t_end = epoch + duration.to(u.s)
        if epoch < t_min or t_end > t_max:
            raise EphemerisOutOfRangeError(
                f"scenario window [{epoch.isot}, {t_end.isot}] "
                f"outside ephemeris coverage [{t_min.isot}, {t_max.isot}]"
            )

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
        """Return (r_km, v_kms) in ICRF barycentric at `epoch`. Shapes: (3,)."""
        target_key, parent_canonical = self._resolver_for(body)
        t = self._ts.tdb_jd(float(epoch.tdb.jd))

        # Find the kernel that has the target.
        target_kernel = self._kernel_with(target_key)
        if target_kernel is None:
            self._raise_missing_kernel(body, target_key, parent_canonical)

        # For parented bodies, the parent kernel must be loaded too — skyfield
        # walks the chain through it to reach SSB. Bail with a useful message
        # if it isn't, before skyfield raises a generic LookupError.
        if parent_canonical is not None:
            parent_target, _ = _SKYFIELD_RESOLVERS[parent_canonical]
            if self._kernel_with(parent_target) is None:
                raise UnknownBodyError(
                    f"body {body!r} resolves through parent {parent_canonical!r} "
                    f"which isn't loaded in any kernel"
                )

        target_pos = target_kernel[target_key].at(t)
        r = np.asarray(target_pos.position.km, dtype=np.float64)
        v = np.asarray(target_pos.velocity.km_per_s, dtype=np.float64)
        return r, v

    def query_many(
        self, body: str | int, epochs: Time,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Vectorized query: return ICRF barycentric (r_km, v_kms) at every
        instant in `epochs`. Shapes: ``(N, 3)``.

        Used by post-hoc analysis (encounter detection) and any other
        code path that needs a full body trajectory at the same N
        sample times. One skyfield evaluation handles the whole batch
        — internally a vectorized SPK lookup that's hundreds of times
        faster than a Python loop of single `.query` calls.
        """
        target_key, parent_canonical = self._resolver_for(body)
        target_kernel = self._kernel_with(target_key)
        if target_kernel is None:
            self._raise_missing_kernel(body, target_key, parent_canonical)
        if parent_canonical is not None:
            parent_target, _ = _SKYFIELD_RESOLVERS[parent_canonical]
            if self._kernel_with(parent_target) is None:
                raise UnknownBodyError(
                    f"body {body!r} resolves through parent {parent_canonical!r} "
                    f"which isn't loaded in any kernel"
                )

        jd = np.asarray(epochs.tdb.jd, dtype=np.float64)
        if jd.ndim == 0:  # promote scalar to length-1 batch
            jd = jd[np.newaxis]
        t = self._ts.tdb_jd(jd)
        target_pos = target_kernel[target_key].at(t)
        # skyfield returns shape (3, N); transpose to (N, 3) for caller-side
        # row-major iteration.
        r = np.asarray(target_pos.position.km, dtype=np.float64).T
        v = np.asarray(target_pos.velocity.km_per_s, dtype=np.float64).T
        return r, v

    def available_bodies(self) -> tuple[str, ...]:
        """Canonical lowercase names of bodies this source can resolve."""
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
        """(t_min, t_max) TDB bounds. Intersection across all loaded kernels."""
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
