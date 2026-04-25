"""Event log schema — one row per sim event (encounter, future Δv, etc.).

For M3, the only event type is `encounter_enter` / `encounter_exit`,
fired when a test particle crosses the Hill sphere of a massive body.
Future milestones (M4 Δv, M4 impact, M5 close-encounter detection at
sub-cadence resolution via heartbeat callback) will reuse this schema.

Persistence: events live alongside the trajectory in a sidecar
parquet file `<basename>.events.parquet` written next to the main
StateHistory parquet. Sidecar rather than embedded so that big runs
(M5 with 1000s of asteroids) don't bloat the main DataFrame and so
post-hoc analyses can append events without rewriting the trajectory.
"""
from __future__ import annotations

from dataclasses import dataclass

EVENT_COLUMNS: tuple[str, ...] = (
    "sample_idx",      # int64 — sample at which the event was detected
    "t_tdb",           # float64 — seconds since scenario.epoch (TDB)
    "kind",            # string — encounter_enter | encounter_exit | delta_v
    "particle",        # string — target particle / body
    "body",            # string — body involved (encounter only; "" for delta_v)
    "distance_km",     # float64 — particle-body separation (encounter only)
    "hill_radius_km",  # float64 — body's Hill radius (encounter only)
    "dv_x_kms",        # float64 — applied Δv x-component (delta_v only)
    "dv_y_kms",        # float64 — applied Δv y-component (delta_v only)
    "dv_z_kms",        # float64 — applied Δv z-component (delta_v only)
)


@dataclass(frozen=True)
class EncounterEvent:
    """One Hill-sphere crossing detected post-hoc on a StateHistory.

    `kind` is "encounter_enter" on the sample where the particle first
    drops below the Hill radius after being outside (or on the first
    sample if it starts inside), and "encounter_exit" on the first
    sample where it leaves. Subsequent re-entries fire fresh events.
    """

    sample_idx: int
    t_tdb: float
    kind: str
    particle: str
    body: str
    distance_km: float
    hill_radius_km: float

    def to_row(self) -> dict[str, object]:
        return {
            "sample_idx": self.sample_idx,
            "t_tdb": self.t_tdb,
            "kind": self.kind,
            "particle": self.particle,
            "body": self.body,
            "distance_km": self.distance_km,
            "hill_radius_km": self.hill_radius_km,
            "dv_x_kms": float("nan"),
            "dv_y_kms": float("nan"),
            "dv_z_kms": float("nan"),
        }


@dataclass(frozen=True)
class DeltaVEvent:
    """One impulsive Δv burn applied during integration.

    Emitted by the runner at the exact burn time (between output
    samples), with `sample_idx` set to the *next* output sample so
    the events sort consistently against encounter rows. The
    distance/hill columns are nan for these rows.
    """

    sample_idx: int
    t_tdb: float
    particle: str
    dv_kms: tuple[float, float, float]

    def to_row(self) -> dict[str, object]:
        return {
            "sample_idx": self.sample_idx,
            "t_tdb": self.t_tdb,
            "kind": "delta_v",
            "particle": self.particle,
            "body": "",
            "distance_km": float("nan"),
            "hill_radius_km": float("nan"),
            "dv_x_kms": float(self.dv_kms[0]),
            "dv_y_kms": float(self.dv_kms[1]),
            "dv_z_kms": float(self.dv_kms[2]),
        }
