"""Generate `scenarios/apophis-2029-flyby.yaml` from JPL Horizons.

Asteroid 99942 Apophis passes within ~32,000 km of Earth's center on
2029-04-13 21:46 UTC — closer than geosynchronous orbit. This is one
of the most-studied close approaches in the asteroid record and the
visual reason ASSIST exists.

The scenario covers a 10-day window (2029-04-09 to 2029-04-19 TDB) at
10-minute cadence, capturing 4 days of approach + 5 days of recede.
Apophis's relative speed at closest approach is ~7.4 km/s, so
10-minute cadence gives ~4,400 km resolution at the crossing — fine
enough to render the geometry of the flyby cleanly in the viewer.

Apophis's IC at the scenario start comes directly from Horizons (the
result of JPL's full N-body propagation from the most-recent orbit
fit up to 2029-04-09). No Kepler propagation, no SBDB drift.

Usage (with the tomcosmos env active and ASSIST kernels fetched):

    python scripts/build_apophis_2029_flyby_scenario.py

Then:

    tomcosmos run scenarios/apophis-2029-flyby.yaml
    tomcosmos view runs/apophis-2029-flyby__*.parquet --follow earth
"""
from __future__ import annotations

from pathlib import Path

from astropy.time import Time

from tomcosmos.targeting import horizons

REPO_ROOT = Path(__file__).resolve().parent.parent
SCENARIO_PATH = REPO_ROOT / "scenarios" / "apophis-2029-flyby.yaml"

_EPOCH_ISO_TDB = "2029-04-09T00:00:00"
_DURATION = "10 day"
_CADENCE = "600 s"  # 10 minutes


_TEMPLATE = """\
schema_version: 1
name: apophis-2029-flyby
# Asteroid 99942 Apophis's 2029-04-13 close approach to Earth — passes
# within ~32,000 km of Earth's center, closer than geosynchronous orbit.
# 10-day window at 10-minute cadence renders the flyby geometry cleanly
# in `tomcosmos view --follow earth`.
#
# Mode A (ASSIST) integration: Apophis is a test particle; gravity from
# the Sun, planets, Moon, and 16 large asteroids comes directly from
# DE440 + sb441-n16.
#
# Requires the ASSIST kernels:
#   tomcosmos fetch-kernels --include assist
# (~770 MB one-time download.)
#
# IC was queried from JPL Horizons at the scenario epoch ({epoch_iso} TDB) —
# Horizons's value is the result of JPL's full N-body propagation from
# the most-recent orbit fit up to that instant. To refresh against the
# latest Horizons orbit fit, re-run scripts/build_apophis_2029_flyby_scenario.py.
epoch: "{epoch_iso} TDB"
duration: "{duration}"
integrator:
  name: ias15
  ephemeris_perturbers: true
output:
  format: parquet
  cadence: "{cadence}"
test_particles:
  - name: apophis
    ic:
      type: explicit
      frame: icrf_barycentric
      r: [{rx!r}, {ry!r}, {rz!r}]
      v: [{vx!r}, {vy!r}, {vz!r}]
"""


def main() -> None:
    epoch = Time(_EPOCH_ISO_TDB, scale="tdb")
    state = horizons.state_at_epoch("99942", epoch)

    text = _TEMPLATE.format(
        epoch_iso=_EPOCH_ISO_TDB,
        duration=_DURATION,
        cadence=_CADENCE,
        rx=float(state.r_km[0]), ry=float(state.r_km[1]), rz=float(state.r_km[2]),
        vx=float(state.v_kms[0]), vy=float(state.v_kms[1]), vz=float(state.v_kms[2]),
    )
    SCENARIO_PATH.write_text(text, encoding="utf-8")
    print(f"wrote {SCENARIO_PATH}")
    print(f"  target_name: {state.target_name}")
    print(f"  epoch: {state.epoch.isot} TDB")
    print(f"  r (km): [{state.r_km[0]:.3e}, {state.r_km[1]:.3e}, {state.r_km[2]:.3e}]")
    print(f"  v (km/s): [{state.v_kms[0]:.4f}, {state.v_kms[1]:.4f}, {state.v_kms[2]:.4f}]")


if __name__ == "__main__":
    main()
