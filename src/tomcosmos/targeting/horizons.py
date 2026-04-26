"""JPL Horizons ingest for Mode A asteroid scenarios.

`state_at_epoch(designation, epoch)` queries JPL Horizons for an
object's ICRF barycentric state vector at a specific instant and
returns `(r_km, v_kms)` ready to drop into a scenario as a
`TestParticleExplicitIc`.

`bulk_states_at_epoch(designations, epoch)` does the same for a
list — Horizons has no body-batched vectors API, so this is just a
sequential loop over `state_at_epoch`. The disk cache (file:
`<cache_dir>/horizons_vectors.json`) makes repeated bulk runs cheap
even for thousands of bodies.

When to use this vs `tomcosmos.targeting.sbdb`:

- **Horizons** is JPL's authoritative propagator. The state vector
  it returns is the result of running JPL's full N-body
  integration from the orbit-determination element-epoch up to
  the queried epoch — so the perturbation history Apophis
  experienced over the last six months is already baked in.
  Use this when you want "the asteroid's actual state at scenario
  epoch" and you're willing to pay ~1.3 s per body for the network
  round-trip (cached after the first hit).

- **SBDB** returns the orbit-determination Keplerian elements at
  their published element-epoch. `sbdb.state_at_epoch` Kepler-
  propagates from there to your scenario epoch using two-body
  math only — which means the perturbation history *isn't* baked
  in. Use this when you want "what JPL knows about this object,
  propagated forward in two-body gravity" — appropriate for
  catalog-scale studies where IC drift is bounded and you're
  studying the dynamics under controlled assumptions, not
  comparing against JPL truth.

Both modules return the same shape — `(r_km, v_kms)` ICRF
barycentric — so the call sites are interchangeable. The library
deliberately does NOT provide a unified "ingest" wrapper that picks
between them with a `source=` flag: the right choice depends on
what question you're answering, and a hidden default would lie.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from astropy import units as u
from astropy.time import Time

from tomcosmos.config import cache_dir as default_cache_dir

_AU_KM = float((1.0 * u.AU).to(u.km).value)
_AU_PER_DAY_TO_KMS = _AU_KM / 86400.0


@dataclass(frozen=True)
class HorizonsState:
    """A single state-vector query result with provenance.

    `target_name` is what Horizons reports — useful for verifying
    the right object came back (designations are sometimes ambiguous
    when comets and asteroids share IAU numbers).
    """

    designation: str
    target_name: str
    epoch: Time
    r_km: np.ndarray
    v_kms: np.ndarray


def state_at_epoch(
    designation: str,
    epoch: Time,
    *,
    cache: Path | None = None,
) -> HorizonsState:
    """Fetch `designation`'s ICRF barycentric state at `epoch` from JPL Horizons.

    `designation` follows JPL conventions: bare numbers ("99942",
    "433"), names ("Apophis"), or provisional ("2024 PT5").

    `cache` is the disk-cache path. Default (`None`) resolves to
    `config.cache_dir() / "horizons_vectors.json"`. Pass an explicit
    `Path` to override (tests use a `tmp_path` fixture). Cache hits
    skip the network round-trip entirely.
    """
    cache_path = cache if cache is not None else (
        default_cache_dir() / "horizons_vectors.json"
    )
    cache_key = _cache_key(designation, epoch)

    cached = _read_cache(cache_path).get(cache_key)
    if cached is not None:
        return _state_from_cache_entry(designation, epoch, cached)

    target_name, r_km, v_kms = _fetch_from_horizons(designation, epoch)

    _write_cache_entry(cache_path, cache_key, {
        "designation": designation,
        "target_name": target_name,
        "epoch_jd_tdb": float(epoch.tdb.jd),
        "r_km": [float(r_km[0]), float(r_km[1]), float(r_km[2])],
        "v_kms": [float(v_kms[0]), float(v_kms[1]), float(v_kms[2])],
    })

    return HorizonsState(
        designation=designation,
        target_name=target_name,
        epoch=epoch,
        r_km=r_km,
        v_kms=v_kms,
    )


def bulk_states_at_epoch(
    designations: list[str],
    epoch: Time,
    *,
    cache: Path | None = None,
) -> list[HorizonsState]:
    """Fetch a list of objects' ICRF barycentric states at `epoch`.

    Horizons has no body-batched vectors API — this is a sequential
    loop. The disk cache keeps re-runs cheap; first run with N new
    bodies costs roughly ~1.3 s × N.

    Returns results in the same order as `designations`. Bodies that
    fail to resolve at JPL surface as RuntimeError on first
    encounter — there's no partial-success mode, since silent gaps
    in the returned list would be a footgun.
    """
    return [state_at_epoch(d, epoch, cache=cache) for d in designations]


# ---------------------------------------------------------------------------
# Cache plumbing
# ---------------------------------------------------------------------------


def _cache_key(designation: str, epoch: Time) -> str:
    """Stable string key for cache lookups.

    Quantize the JD to 6 decimal places (microseconds) so floating-
    point representation noise across query times doesn't fragment
    the cache. Anyone querying within 1 µs of a previous query
    really does want the same answer.
    """
    return f"{designation}|{float(epoch.tdb.jd):.6f}"


def _read_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def _write_cache_entry(
    path: Path, key: str, entry: dict[str, Any],
) -> None:
    """Append-only update to the JSON cache, with atomic write so
    concurrent test workers don't corrupt the file. The lock-free
    last-write-wins is acceptable for a deterministic cache: any
    overwriting writer is producing the identical entry anyway."""
    cache = _read_cache(path)
    cache[key] = entry
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, sort_keys=True)
        f.write("\n")
    tmp.replace(path)


def _state_from_cache_entry(
    designation: str, epoch: Time, entry: dict[str, Any],
) -> HorizonsState:
    return HorizonsState(
        designation=designation,
        target_name=str(entry.get("target_name", designation)),
        epoch=epoch,
        r_km=np.asarray(entry["r_km"], dtype=np.float64),
        v_kms=np.asarray(entry["v_kms"], dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# Horizons network call
# ---------------------------------------------------------------------------


def _fetch_from_horizons(
    designation: str, epoch: Time,
) -> tuple[str, np.ndarray, np.ndarray]:
    """Single Horizons round-trip via astroquery.

    `location="@0"` → Solar System Barycenter origin.
    `refplane="earth"` → ICRF (J2000 mean equator), matching the
    frame tomcosmos uses internally and what
    `state.frames.ecliptic_to_icrf` produces.
    """
    from astroquery.jplhorizons import Horizons

    h = Horizons(
        id=designation, id_type="smallbody",
        location="@0",
        epochs=float(epoch.tdb.jd),
    )
    try:
        table = h.vectors(refplane="earth")
    except Exception as e:  # noqa: BLE001 — astroquery raises a zoo of types
        raise RuntimeError(
            f"Horizons query for {designation!r} at JD {epoch.tdb.jd} failed: {e}"
        ) from e

    if len(table) != 1:
        raise RuntimeError(
            f"Horizons returned {len(table)} rows for {designation!r}; expected 1"
        )

    target_name = str(table["targetname"][0])
    r_km = np.array([
        float(table["x"][0]),
        float(table["y"][0]),
        float(table["z"][0]),
    ]) * _AU_KM
    v_kms = np.array([
        float(table["vx"][0]),
        float(table["vy"][0]),
        float(table["vz"][0]),
    ]) * _AU_PER_DAY_TO_KMS
    return target_name, r_km, v_kms
