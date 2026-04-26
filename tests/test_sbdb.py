"""M5a — SBDB ingest + Mode A round-trip against JPL Horizons.

These tests are the architectural proof that tomcosmos's Mode A
integrator agrees with JPL Horizons (the same physics: DE440 + 16
asteroid perturbers + GR + J2) for a real, well-known asteroid.
Apophis is the canonical NEO test case — historically observed,
arc spans 20+ years, and famous for its 2029-04-13 close approach
that the orbit-determination community has scrutinized exhaustively.

Marked `assist` because Mode A needs the full DE440 + sb441-n16
kernels (~~750 MB), and `ephemeris` because they hit JPL's
SBDB / Horizons web services — both must be reachable on the
test machine.
"""
from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u
from astropy.time import Time

from tomcosmos.config import assist_asteroid_kernel, assist_planet_kernel
from tomcosmos.state.ephemeris import EphemerisSource

pytestmark = [pytest.mark.assist, pytest.mark.ephemeris]

AU_KM = 1.495978707e8
AU_PER_DAY_TO_KMS = AU_KM / 86400.0


def _require_assist_kernels() -> None:
    if not assist_planet_kernel().exists() or not assist_asteroid_kernel().exists():
        pytest.skip(
            "ASSIST kernels not present; run "
            "`tomcosmos fetch-kernels --include assist`"
        )


def _horizons_state_km_kms(designation: str, epoch: Time) -> tuple[np.ndarray, np.ndarray]:
    """Query JPL Horizons for `designation` at `epoch`, return ICRF
    barycentric (r_km, v_kms) at 64-bit precision."""
    pytest.importorskip("astroquery.jplhorizons")
    from astroquery.jplhorizons import Horizons

    h = Horizons(
        id=designation, id_type="smallbody",
        location="@0",   # Solar System Barycenter
        epochs=float(epoch.tdb.jd),
    )
    # refplane="earth" → ICRF / J2000 mean equator (the same frame
    # tomcosmos uses internally; matches our SBDB ingest's ecliptic→ICRF
    # rotation output).
    state = h.vectors(refplane="earth")
    r = np.array([
        float(state["x"][0]), float(state["y"][0]), float(state["z"][0]),
    ]) * AU_KM
    v = np.array([
        float(state["vx"][0]), float(state["vy"][0]), float(state["vz"][0]),
    ]) * AU_PER_DAY_TO_KMS
    return r, v


# --- IC seeding sanity ------------------------------------------------------


def test_apophis_state_at_element_epoch_matches_horizons(
    ephemeris_source: EphemerisSource,
) -> None:
    """Bootstrap test: at the SBDB element epoch our Kepler-from-elements
    result must match Horizons's published state at the same instant.
    The only residual is float64 roundoff between two ways of computing
    the same physics; expect tens of meters, definitely under 1 km."""
    pytest.importorskip("astroquery.jplsbdb")
    from tomcosmos.targeting import sbdb

    orbit = sbdb.query("99942")
    r_ours, v_ours = sbdb.state_at_epoch(orbit, orbit.elements_epoch, ephemeris_source)
    r_horizons, v_horizons = _horizons_state_km_kms("99942", orbit.elements_epoch)

    dr = float(np.linalg.norm(r_ours - r_horizons))
    dv = float(np.linalg.norm(v_ours - v_horizons))
    assert dr < 1.0, f"position disagreement at element epoch: {dr:.3f} km"
    assert dv < 1e-5, f"velocity disagreement at element epoch: {dv:.6e} km/s"


# --- Mode A round-trip ------------------------------------------------------


def test_apophis_30_day_propagation_in_mode_a_matches_horizons(
    ephemeris_source: EphemerisSource,
) -> None:
    """The headline M5a test: integrate Apophis in Mode A for 30 days,
    compare end state to Horizons. Same force model on both sides
    (DE440 + sb441-n16 + GR + J2), so this is a tight check.

    The scenario starts at the SBDB element epoch — no Kepler-only
    propagation between SBDB's "elements valid here" and the
    integration start. That isolates Mode A's integration error
    from IC drift; if scenario.epoch were months past the element
    epoch, pure-Kepler IC seeding would inject perturbation drift
    Mode A then faithfully reproduces, and the test would be
    measuring the wrong thing.

    For "I want to start at an arbitrary epoch with the
    best-known state," real users should query Horizons directly
    rather than Kepler-propagating SBDB elements over months.
    Tracked as future work.
    """
    _require_assist_kernels()
    pytest.importorskip("assist")
    pytest.importorskip("astroquery.jplsbdb")

    from tomcosmos import Scenario, run
    from tomcosmos.targeting import sbdb

    orbit = sbdb.query("99942")
    epoch = orbit.elements_epoch
    duration_days = 30.0

    r0_km, v0_kms = sbdb.state_at_epoch(orbit, epoch, ephemeris_source)

    scenario = Scenario.model_validate({
        "schema_version": 1,
        "name": "apophis-30day-roundtrip",
        "epoch": f"{epoch.isot} TDB",
        "duration": f"{duration_days} day",
        "integrator": {"name": "ias15", "ephemeris_perturbers": True},
        "output": {"format": "parquet", "cadence": "1 day"},
        "test_particles": [{
            "name": "apophis",
            "ic": {
                "type": "explicit",
                "r": [float(r0_km[0]), float(r0_km[1]), float(r0_km[2])],
                "v": [float(v0_kms[0]), float(v0_kms[1]), float(v0_kms[2])],
                "frame": "icrf_barycentric",
            },
        }],
    })

    history = run(scenario, source=ephemeris_source, allow_dirty=True)
    end_row = history.body_trajectory("apophis").iloc[-1]
    r_end_ours = np.array([end_row["x"], end_row["y"], end_row["z"]])
    v_end_ours = np.array([end_row["vx"], end_row["vy"], end_row["vz"]])

    end_epoch = epoch + duration_days * u.day
    r_end_horizons, v_end_horizons = _horizons_state_km_kms("99942", end_epoch)

    dr_km = float(np.linalg.norm(r_end_ours - r_end_horizons))
    dv_kms = float(np.linalg.norm(v_end_ours - v_end_horizons))

    # Same physics on both sides (DE440 + 16 asteroid perturbers + GR + J2).
    # Residual is integrator step size on tomcosmos's side + roundoff. Tens
    # of km is realistic for IAS15 over 30 days against Horizons's
    # higher-order propagator. This bound is generous to avoid flakes; if
    # we see <1 km regularly we'll tighten.
    assert dr_km < 100.0, (
        f"30-day Mode A position disagreement vs Horizons: {dr_km:.3f} km"
    )
    assert dv_kms < 1e-3, (
        f"30-day Mode A velocity disagreement vs Horizons: {dv_kms:.6e} km/s"
    )


def test_apophis_returns_to_kepler_baseline_over_30_days(
    ephemeris_source: EphemerisSource,
) -> None:
    """Sanity: over 30 days, planetary perturbations on Apophis are small
    but nonzero. tomcosmos in Mode A should differ from a pure-Kepler
    propagation by enough to be detectable (perturbations exist) but
    not by so much that something is broken.

    Acts as a regression: if Mode A's force loop ever loses ASSIST's
    perturber set, this comparison would suddenly drop to ~0 and the
    headline test above would fail in step.
    """
    _require_assist_kernels()
    pytest.importorskip("assist")
    pytest.importorskip("astroquery.jplsbdb")

    from tomcosmos import Scenario, run
    from tomcosmos.targeting import sbdb

    orbit = sbdb.query("99942")
    epoch = orbit.elements_epoch
    duration_days = 30.0

    r0_km, v0_kms = sbdb.state_at_epoch(orbit, epoch, ephemeris_source)
    end_epoch = epoch + duration_days * u.day
    r_pure_kepler, _ = sbdb.state_at_epoch(orbit, end_epoch, ephemeris_source)

    scenario = Scenario.model_validate({
        "schema_version": 1,
        "name": "apophis-30day-baseline",
        "epoch": f"{epoch.isot} TDB",
        "duration": f"{duration_days} day",
        "integrator": {"name": "ias15", "ephemeris_perturbers": True},
        "output": {"format": "parquet", "cadence": "1 day"},
        "test_particles": [{
            "name": "apophis",
            "ic": {
                "type": "explicit",
                "r": [float(r0_km[0]), float(r0_km[1]), float(r0_km[2])],
                "v": [float(v0_kms[0]), float(v0_kms[1]), float(v0_kms[2])],
                "frame": "icrf_barycentric",
            },
        }],
    })

    history = run(scenario, source=ephemeris_source, allow_dirty=True)
    end_row = history.body_trajectory("apophis").iloc[-1]
    r_end_assist = np.array([end_row["x"], end_row["y"], end_row["z"]])

    perturbation_km = float(np.linalg.norm(r_end_assist - r_pure_kepler))
    # Apophis is an NEO; planetary tugs over 30 days move it tens to
    # hundreds of km from the pure-Kepler baseline. Anything inside
    # [10 km, 50_000 km] is "perturbations are working." Outside that
    # range suggests either no perturbations (low end) or numerical
    # blowup (high end).
    assert 10.0 < perturbation_km < 50_000.0, (
        f"perturbation magnitude {perturbation_km:.1f} km outside the "
        "expected band — Mode A force loop may be miswired"
    )
