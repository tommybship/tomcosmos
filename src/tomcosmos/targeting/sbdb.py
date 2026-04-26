"""JPL Small-Body Database (SBDB) ingest for Mode A asteroid scenarios.

`query(designation)` returns an `SBDBOrbit` carrying the asteroid's
osculating Keplerian elements at SBDB's published element-epoch.
`state_at_epoch(orbit, target_epoch, source)` Kepler-propagates those
elements forward (or backward) to `target_epoch` and returns ICRF
barycentric `(r_km, v_kms)` ready to drop into a scenario as a
`TestParticleExplicitIc`.

Mode A (REBOUND + ASSIST) is the natural integrator for SBDB-ingested
asteroids: ASSIST's force loop folds in the Sun + planets + Moon +
16-asteroid perturbers from DE440 / sb441-n16 directly. JPL Horizons
runs the same physics to publish ephemerides; tomcosmos's Mode A
output should agree with Horizons to ASSIST's truncation precision
(sub-km over months for typical asteroids).

The Kepler-propagation step here is **two-body only** — heliocentric
gravity, no perturbers. It exists solely to bridge the gap between
SBDB's element-epoch and the scenario's epoch when those don't
coincide. Once the IC lands inside Mode A, ASSIST's full N-body
takes over. Don't use `state_at_epoch` for "real" propagation;
that's the integrator's job.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from astropy import units as u
from astropy.time import Time

from tomcosmos.state.ephemeris import EphemerisSource
from tomcosmos.state.frames import ecliptic_to_icrf
from tomcosmos.state.kepler import keplerian_to_state

# Heliocentric Sun GM in km³/s², matching tomcosmos.targeting.transfer.
MU_SUN_KM3_S2 = 1.32712440018e11

_AU_KM = float((1.0 * u.AU).to(u.km).value)


@dataclass(frozen=True)
class SBDBOrbit:
    """Osculating Keplerian elements pulled from JPL SBDB.

    Heliocentric, J2000 ecliptic frame — that's how SBDB publishes
    them. `elements_epoch` is the TDB instant the elements are valid
    at (typically the body's most recent orbit-determination epoch,
    usually months to a couple years stale). `state_at_epoch`
    propagates two-body Kepler from this epoch to whatever the
    scenario needs.
    """

    designation: str
    fullname: str
    elements_epoch: Time            # TDB
    a_km: float
    e: float
    inc_deg: float
    raan_deg: float
    argp_deg: float
    mean_anom_deg: float


def query(designation: str) -> SBDBOrbit:
    """Query JPL SBDB for `designation` and return its Keplerian elements.

    `designation` follows JPL conventions — bare numbers ("99942",
    "433"), names ("Apophis", "Eros"), or provisional designations
    ("2024 PT5"). Anything `astroquery.jplsbdb.SBDB` accepts.

    Network call. Raises `RuntimeError` if SBDB returns an error or
    the response is missing fields tomcosmos depends on.
    """
    from astroquery.jplsbdb import SBDB

    raw = SBDB.query(designation, full_precision=True)
    if "object" not in raw or "orbit" not in raw:
        raise RuntimeError(
            f"SBDB returned an unexpected payload for {designation!r}: "
            f"keys={list(raw.keys())}"
        )

    obj = raw["object"]
    orbit = raw["orbit"]
    elem = orbit["elements"]

    epoch_jd = _scalar_value(orbit["epoch"], u.day)

    return SBDBOrbit(
        designation=designation,
        fullname=str(obj.get("fullname", designation)),
        elements_epoch=Time(epoch_jd, format="jd", scale="tdb"),
        a_km=_scalar_value(elem["a"], u.km),
        e=_scalar_value(elem["e"], u.dimensionless_unscaled),
        inc_deg=_scalar_value(elem["i"], u.deg),
        raan_deg=_scalar_value(elem["om"], u.deg),
        argp_deg=_scalar_value(elem["w"], u.deg),
        mean_anom_deg=_scalar_value(elem["ma"], u.deg),
    )


def state_at_epoch(
    orbit: SBDBOrbit,
    target_epoch: Time,
    source: EphemerisSource,
) -> tuple[np.ndarray, np.ndarray]:
    """Kepler-propagate the orbit from its element-epoch to `target_epoch`,
    return ICRF barycentric `(r_km, v_kms)`.

    Two-body math against the Sun's GM, then ecliptic → ICRF
    rotation, then add the Sun's barycentric state from `source` at
    `target_epoch` to lift heliocentric → barycentric.

    Accuracy of the propagation step alone is ~10s of km over a year
    for a typical main-belt asteroid (the missing physics is
    planetary perturbations, which Mode A's integrator picks up
    once you hand it this state vector). For epochs close to
    `orbit.elements_epoch` the residual is float-precision noise.
    """
    dt_s = float((target_epoch - orbit.elements_epoch).to(u.s).value)

    r_helio_ecliptic, v_helio_ecliptic = keplerian_to_state(
        a=orbit.a_km,
        e=orbit.e,
        inc_rad=np.deg2rad(orbit.inc_deg),
        raan_rad=np.deg2rad(orbit.raan_deg),
        argp_rad=np.deg2rad(orbit.argp_deg),
        mean_anom_rad=np.deg2rad(orbit.mean_anom_deg),
        mu=MU_SUN_KM3_S2,
        dt=dt_s,
    )

    r_helio_icrf = ecliptic_to_icrf(r_helio_ecliptic)
    v_helio_icrf = ecliptic_to_icrf(v_helio_ecliptic)

    r_sun, v_sun = source.query("sun", target_epoch)
    return r_helio_icrf + r_sun, v_helio_icrf + v_sun


def bulk_states_at_epoch(
    designations: list[str],
    target_epoch: Time,
    source: EphemerisSource,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Bulk variant: SBDB-query each designation, Kepler-propagate to
    `target_epoch`, return `(target_name, r_km, v_kms)` tuples in
    request order.

    `target_name` is what SBDB's `object.fullname` reports — useful
    for the scenario-builder code so the test particle's name reflects
    JPL's canonical form (e.g. "99942 Apophis (2004 MN4)") rather
    than whatever shorthand the user typed.

    No bulk SBDB endpoint exists for orbit elements, so this is a
    sequential `query` loop — but SBDB queries are cheap (<200 ms
    each) and don't require a disk cache to be tractable for
    population studies. For 1,000 NEOs expect ~3 minutes.
    """
    out: list[tuple[str, np.ndarray, np.ndarray]] = []
    for designation in designations:
        orbit = query(designation)
        r_km, v_kms = state_at_epoch(orbit, target_epoch, source)
        out.append((orbit.fullname, r_km, v_kms))
    return out


def _scalar_value(quantity: Any, expected_unit: u.Unit) -> float:
    """Convert SBDB's astropy.Quantity (or bare float for dimensionless
    fields like eccentricity) into a plain float in `expected_unit`."""
    if hasattr(quantity, "to"):
        return float(quantity.to(expected_unit).value)
    return float(quantity)
