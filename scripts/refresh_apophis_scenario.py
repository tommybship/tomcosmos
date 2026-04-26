"""Regenerate `scenarios/apophis-30day.yaml` from the current JPL SBDB
orbit-determination for asteroid 99942 Apophis.

JPL re-fits small-body orbits as new astrometric observations come in
(roughly monthly for Apophis). The scenario YAML carries a snapshot of
the state vector at the element-set epoch, so re-running this script
brings the committed scenario in sync with whatever Apophis's orbit
looks like *today*.

Usage (with the tomcosmos env active):

    python scripts/refresh_apophis_scenario.py

Writes `scenarios/apophis-30day.yaml`. The diff is the meaningful part
— it shows how much Apophis's published orbit shifted since the last
refresh.
"""
from __future__ import annotations

from pathlib import Path

from tomcosmos.state.ephemeris import EphemerisSource
from tomcosmos.targeting import sbdb

REPO_ROOT = Path(__file__).resolve().parent.parent
SCENARIO_PATH = REPO_ROOT / "scenarios" / "apophis-30day.yaml"


_TEMPLATE = """\
schema_version: 1
name: apophis-30day
# 30-day Mode A propagation of asteroid 99942 Apophis. Uses ASSIST's
# DE440 + sb441-n16 force loop directly — same physics JPL Horizons
# uses to publish small-body ephemerides — so this run agrees with
# Horizons to < 100 m over 30 days (verified in tests/test_sbdb.py).
#
# Requires the ASSIST kernels:
#   tomcosmos fetch-kernels --include assist
# (~770 MB one-time download; full DE440 + sb441-n16.)
#
# IC was snapshotted from JPL SBDB at the orbit-determination
# element epoch ({epoch_iso} TDB). The state vector is heliocentric
# Keplerian elements converted to ICRF barycentric (r_km, v_kms)
# via tomcosmos.targeting.sbdb. To refresh against the latest
# orbit determination — JPL re-fits asteroid orbits as new
# observations land — re-run scripts/refresh_apophis_scenario.py
# and it overwrites the values below.
epoch: "{epoch_iso} TDB"
duration: "30 day"
integrator:
  name: ias15
  ephemeris_perturbers: true
output:
  format: parquet
  cadence: "1 day"
test_particles:
  - name: apophis
    ic:
      type: explicit
      frame: icrf_barycentric
      r: [{rx!r}, {ry!r}, {rz!r}]
      v: [{vx!r}, {vy!r}, {vz!r}]
"""


def main() -> None:
    source = EphemerisSource()
    orbit = sbdb.query("99942")
    r, v = sbdb.state_at_epoch(orbit, orbit.elements_epoch, source)

    text = _TEMPLATE.format(
        epoch_iso=orbit.elements_epoch.isot,
        rx=float(r[0]), ry=float(r[1]), rz=float(r[2]),
        vx=float(v[0]), vy=float(v[1]), vz=float(v[2]),
    )
    SCENARIO_PATH.write_text(text, encoding="utf-8")
    print(f"wrote {SCENARIO_PATH}")
    print(f"  fullname: {orbit.fullname}")
    print(f"  element epoch: {orbit.elements_epoch.isot} TDB")
    print(f"  a={orbit.a_km:.0f} km  e={orbit.e:.6f}  i={orbit.inc_deg:.4f} deg")


if __name__ == "__main__":
    main()
