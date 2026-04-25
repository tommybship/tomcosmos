"""Hill-sphere encounter detection.

Walks a `StateHistory` post-hoc and emits `EncounterEvent`s every time
a test particle (or any non-Sun body) crosses the Hill radius of a
massive body. Output cadence sets the temporal resolution: events are
flagged at sample granularity. Fast flybys that occur entirely between
two output samples are missed; M5's heartbeat callback adds sub-cadence
detection if and when it's needed.

Hill radius for a body orbiting the Sun is the standard
    r_H ≈ a · (m / (3 M_☉))^(1/3)
where `a` is the body's semi-major axis. We approximate `a` by the
sun-relative distance at sample 0 — close to the true semi-major axis
for low-eccentricity bodies (planets), and the M3 test scenarios all
fit that profile. A future improvement would compute osculating
elements per-sample and use the time-varying r_H, but the simple
constant-per-body approach is what's needed for tadpole demos and
asteroid encounters.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from tomcosmos.constants import resolve_body_constant
from tomcosmos.exceptions import UnknownBodyError
from tomcosmos.io.history import StateHistory
from tomcosmos.state.events import EVENT_COLUMNS, EncounterEvent


def detect_hill_encounters(history: StateHistory) -> pd.DataFrame:
    """Return one row per encounter enter/exit found in `history`.

    Pairs every test particle with every massive body in the run
    (test particles are scenario-declared; massive bodies are everyone
    else *except* the Sun, which has no Hill sphere of its own here).
    For each pair, walks the per-sample distance and flags transitions
    across the body's Hill radius.
    """
    if history.df.empty:
        return _empty_events_df()

    test_particle_names = _test_particle_names(history)
    massive_bodies = _massive_body_names(history)

    if not test_particle_names or not massive_bodies:
        return _empty_events_df()

    # Pre-compute per-body trajectories as (n_samples, 3) arrays for
    # vectorized distance math. The pivot also exposes t_tdb for stamps.
    pivot = (
        history.df.sort_values(["sample_idx", "body"])
        .pivot(index="sample_idx", columns="body", values=["x", "y", "z", "t_tdb"])
    )
    t_tdb = pivot["t_tdb"].iloc[:, 0].to_numpy(dtype=np.float64)

    def _xyz(name: str) -> np.ndarray:
        return np.stack(
            [pivot[("x", name)], pivot[("y", name)], pivot[("z", name)]],
            axis=-1,
        )

    sun_pos = _xyz("sun") if "sun" in history.df["body"].unique() else None

    rows: list[dict[str, object]] = []
    for body_name in massive_bodies:
        r_H = _hill_radius_km(history, body_name, sun_pos)
        if r_H is None:
            continue  # no mass available — skip silently
        body_xyz = _xyz(body_name)
        for tp_name in test_particle_names:
            tp_xyz = _xyz(tp_name)
            d = np.linalg.norm(tp_xyz - body_xyz, axis=1)
            for ev in _crossings(d, r_H, body_name, tp_name, t_tdb):
                rows.append(ev.to_row())

    if not rows:
        return _empty_events_df()
    df = pd.DataFrame.from_records(rows, columns=list(EVENT_COLUMNS))
    return df.sort_values(["sample_idx", "particle", "body"]).reset_index(drop=True)


def _crossings(
    distance: np.ndarray,
    r_H: float,
    body: str,
    particle: str,
    t_tdb: np.ndarray,
) -> list[EncounterEvent]:
    """Walk per-sample distance vs Hill radius and emit edge events.

    First sample inside emits an `encounter_enter`; the first sample
    that lifts back outside emits `encounter_exit`. Re-entries fire a
    fresh enter event. No hysteresis — output cadence is the natural
    smoothing, and a particle hovering exactly at the Hill boundary
    deserves to be flagged.
    """
    inside = distance < r_H
    events: list[EncounterEvent] = []
    if len(inside) == 0:
        return events
    if inside[0]:
        events.append(EncounterEvent(
            sample_idx=0, t_tdb=float(t_tdb[0]),
            kind="encounter_enter",
            particle=particle, body=body,
            distance_km=float(distance[0]),
            hill_radius_km=r_H,
        ))
    transitions = np.diff(inside.astype(np.int8))
    for idx in np.where(transitions != 0)[0]:
        i = int(idx) + 1  # transitions[i-1] = inside[i] - inside[i-1]
        kind = "encounter_enter" if inside[i] else "encounter_exit"
        events.append(EncounterEvent(
            sample_idx=i, t_tdb=float(t_tdb[i]),
            kind=kind,
            particle=particle, body=body,
            distance_km=float(distance[i]),
            hill_radius_km=r_H,
        ))
    return events


def _hill_radius_km(
    history: StateHistory, body_name: str, sun_pos: np.ndarray | None,
) -> float | None:
    """r_H = a · (m / (3 M_sun))^(1/3) using sample-0 sun-relative
    distance as `a`. Returns None when the body's mass isn't available
    (test particles get no Hill radius — they can't *be* the body whose
    Hill sphere is being checked, only the particle entering one)."""
    try:
        const = resolve_body_constant(body_name)
        m_kg = const.mass_kg
    except UnknownBodyError:
        return None

    sun_const_mass: float
    try:
        sun_const_mass = resolve_body_constant("sun").mass_kg
    except UnknownBodyError:  # pragma: no cover
        sun_const_mass = 1.989e30

    body_pos = history.df[history.df["body"] == body_name][
        ["sample_idx", "x", "y", "z"]
    ].sort_values("sample_idx")
    if body_pos.empty:
        return None
    r0 = body_pos[["x", "y", "z"]].iloc[0].to_numpy(dtype=np.float64)
    a_km: float
    if sun_pos is not None:
        a_km = float(np.linalg.norm(r0 - sun_pos[0]))
    else:
        # No Sun in scenario — fall back to barycentric distance. M2c
        # earth-moon (no sun-as-perturber, just sun + earth + moon) still
        # has the sun and uses this branch.
        a_km = float(np.linalg.norm(r0))
    if a_km <= 0.0:
        return None
    return a_km * (m_kg / (3.0 * sun_const_mass)) ** (1.0 / 3.0)


def _test_particle_names(history: StateHistory) -> list[str]:
    return [p.name for p in history.scenario.test_particles]


def _massive_body_names(history: StateHistory) -> list[str]:
    """Massive bodies a test particle might have a Hill encounter with.

    Excludes the Sun (its 'Hill sphere' is the heliopause — meaningless
    in this context) and excludes the test particles themselves.
    """
    tp_names = set(_test_particle_names(history))
    return [
        b.name for b in history.scenario.bodies
        if b.name.lower() != "sun" and b.name not in tp_names
    ]


def _empty_events_df() -> pd.DataFrame:
    return pd.DataFrame(
        {col: pd.Series(dtype=_dtype_for(col)) for col in EVENT_COLUMNS}
    )


def _dtype_for(col: str) -> str:
    return {
        "sample_idx": "int64",
        "t_tdb": "float64",
        "kind": "string",
        "particle": "string",
        "body": "string",
        "distance_km": "float64",
        "hill_radius_km": "float64",
    }[col]
