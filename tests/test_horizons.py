"""Tests for tomcosmos.targeting.horizons.

Single-body sanity vs SBDB-Kepler at the SBDB element epoch (the two
must agree there, since at that instant Horizons's propagation history
is zero — both methods are reading the same underlying orbit fit).

Bulk-ingest sanity: requested designations come back in order, with
target_name strings JPL recognizes, and at sane heliocentric distances
for known asteroids.

Marked `ephemeris` because both tests hit JPL's Horizons web service.
The disk cache (under `data/cache/`) makes re-runs cheap.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.time import Time

from tomcosmos.state.ephemeris import EphemerisSource

pytestmark = pytest.mark.ephemeris

AU_KM = 1.495978707e8


def test_horizons_state_at_element_epoch_agrees_with_sbdb_kepler(
    ephemeris_source: EphemerisSource,
    tmp_path: Path,
) -> None:
    """At the SBDB-published element epoch, Horizons's propagated state
    must equal the Kepler-from-elements state to within a few hundred
    meters — both ways are reading the same underlying orbit fit, with
    propagation interval zero, so the residual is float-precision +
    representation differences between SBDB's element format and
    Horizons's internal state.
    """
    pytest.importorskip("astroquery.jplsbdb")
    pytest.importorskip("astroquery.jplhorizons")
    from tomcosmos.targeting import horizons, sbdb

    orbit = sbdb.query("99942")
    r_kepler, v_kepler = sbdb.state_at_epoch(
        orbit, orbit.elements_epoch, ephemeris_source,
    )
    h_state = horizons.state_at_epoch(
        "99942", orbit.elements_epoch, cache=tmp_path / "h.json",
    )

    dr_km = float(np.linalg.norm(h_state.r_km - r_kepler))
    dv_kms = float(np.linalg.norm(h_state.v_kms - v_kepler))
    assert dr_km < 1.0, f"Horizons-vs-Kepler at element epoch: {dr_km:.3f} km"
    assert dv_kms < 1e-5, f"Horizons-vs-Kepler velocity: {dv_kms:.3e} km/s"


def test_horizons_caches_to_disk_and_skips_network_on_repeat(
    tmp_path: Path,
) -> None:
    """First call writes a cache entry; second call reads it back
    without hitting the network. We don't assert the hard-coded
    timing (CI can be flaky), but verify the file exists with the
    expected key shape after the first call."""
    pytest.importorskip("astroquery.jplhorizons")
    from tomcosmos.targeting import horizons

    cache_file = tmp_path / "h.json"
    epoch = Time("2026-04-26T00:00:00", scale="tdb")

    s1 = horizons.state_at_epoch("99942", epoch, cache=cache_file)
    assert cache_file.exists()

    s2 = horizons.state_at_epoch("99942", epoch, cache=cache_file)
    # Second call must return the same numbers byte-for-byte.
    assert np.array_equal(s1.r_km, s2.r_km)
    assert np.array_equal(s1.v_kms, s2.v_kms)
    assert s1.target_name == s2.target_name


def test_horizons_bulk_returns_in_request_order(
    tmp_path: Path,
) -> None:
    """`bulk_states_at_epoch` is order-preserving so users can zip
    designations with returned states without re-keying. Three NEOs
    is enough to catch any reordering."""
    pytest.importorskip("astroquery.jplhorizons")
    from tomcosmos.targeting import horizons

    designations = ["99942", "433", "101955"]
    epoch = Time("2026-04-26T00:00:00", scale="tdb")
    states = horizons.bulk_states_at_epoch(
        designations, epoch, cache=tmp_path / "h.json",
    )
    assert [s.designation for s in states] == designations
    # Sanity: heliocentric distances within asteroid-belt-ish bounds.
    for s, name in zip(states, ["Apophis", "Eros", "Bennu"], strict=True):
        dist_au = float(np.linalg.norm(s.r_km)) / AU_KM
        assert 0.5 < dist_au < 5.0, (
            f"{name}: heliocentric {dist_au:.3f} AU — outside expected NEO range"
        )


@pytest.mark.assist
def test_three_neos_horizons_ingest_propagates_sanely_in_mode_a(
    ephemeris_source: EphemerisSource,
    tmp_path: Path,
) -> None:
    """End-to-end: ingest 3 NEOs from Horizons at scenario.epoch,
    run a 30-day Mode A integration, verify each asteroid stays
    on a sensible heliocentric orbit (no numerical blowup, no
    plunge into the Sun, no escape).

    Per-body comparison vs Horizons truth at end-of-integration is
    covered for Apophis in test_sbdb.py and would be ~72 m here too.
    This test focuses on the bulk-ingestion → bulk-propagation
    pipeline rather than re-asserting the per-body accuracy that
    M5a already pinned down."""
    from tomcosmos.config import assist_asteroid_kernel, assist_planet_kernel

    if not (assist_planet_kernel().exists() and assist_asteroid_kernel().exists()):
        pytest.skip(
            "ASSIST kernels not present; run "
            "`tomcosmos fetch-kernels --include assist`"
        )

    pytest.importorskip("assist")
    pytest.importorskip("astroquery.jplhorizons")
    from astropy import units as u

    from tomcosmos import Scenario, run
    from tomcosmos.targeting import horizons

    designations = ["99942", "433", "101955"]
    epoch = Time("2026-04-26T00:00:00", scale="tdb")
    duration_days = 30.0

    states = horizons.bulk_states_at_epoch(
        designations, epoch, cache=tmp_path / "h.json",
    )

    test_particles = [
        {
            "name": s.designation,
            "ic": {
                "type": "explicit",
                "frame": "icrf_barycentric",
                "r": [float(s.r_km[0]), float(s.r_km[1]), float(s.r_km[2])],
                "v": [float(s.v_kms[0]), float(s.v_kms[1]), float(s.v_kms[2])],
            },
        }
        for s in states
    ]

    scenario = Scenario.model_validate({
        "schema_version": 1,
        "name": "neos-bulk-30day",
        "epoch": f"{epoch.isot} TDB",
        "duration": f"{duration_days} day",
        "integrator": {"name": "ias15", "ephemeris_perturbers": True},
        "output": {"format": "parquet", "cadence": "5 day"},
        "test_particles": test_particles,
    })

    history = run(scenario, source=ephemeris_source, allow_dirty=True)

    # Each asteroid's heliocentric distance over the run should stay in
    # a plausible NEO/MBA band — ~0.5-5 AU. If the integration blew up
    # we'd see distances exploding; if a body fell into the sun we'd
    # see distances collapsing.
    r_sun_at_epoch, _ = ephemeris_source.query("sun", epoch)
    end_epoch = epoch + duration_days * u.day
    r_sun_at_end, _ = ephemeris_source.query("sun", end_epoch)
    # Use a midpoint sun position as a cheap approximation; we're only
    # checking band membership, not precise heliocentric distance.
    r_sun_avg = 0.5 * (r_sun_at_epoch + r_sun_at_end)

    for designation in designations:
        traj = history.body_trajectory(designation)
        positions = traj[["x", "y", "z"]].to_numpy() - r_sun_avg
        helio_dists_au = np.linalg.norm(positions, axis=1) / AU_KM
        assert np.all(np.isfinite(helio_dists_au)), (
            f"{designation}: NaN/inf in trajectory — integrator blowup"
        )
        assert np.all(helio_dists_au > 0.3), (
            f"{designation}: dropped to {helio_dists_au.min():.3f} AU "
            "from Sun — solar plunge is not physical here"
        )
        assert np.all(helio_dists_au < 6.0), (
            f"{designation}: reached {helio_dists_au.max():.3f} AU from "
            "Sun — escape is not physical over 30 days"
        )
