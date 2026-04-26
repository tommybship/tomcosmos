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

In Mode A (`integrator.ephemeris_perturbers=True`), `scenario.bodies`
is empty by construction — major-body gravity comes from ASSIST's
DE440 / sb441-n16 kernels at integration time, not from declared
particles. The detector pulls those bodies' trajectories back from
the supplied `EphemerisSource` for post-hoc encounter math, so an
NEO test particle's flyby of Earth still produces an encounter event
even though Earth was never an explicit `Body` in the scenario.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from astropy import units as u

from tomcosmos.constants import resolve_body_constant
from tomcosmos.exceptions import UnknownBodyError
from tomcosmos.io.history import StateHistory
from tomcosmos.state.ephemeris import EphemerisSource
from tomcosmos.state.events import EVENT_COLUMNS, EncounterEvent

# Bodies whose Hill spheres tomcosmos checks in Mode A. The Sun is
# omitted (its "Hill sphere" is the heliopause, not a useful encounter
# threshold for asteroid analysis). Pluto is omitted because its system
# barycenter motion in DE440s confuses the simple sample-0 distance
# proxy for `a`; add it back if a Pluto-flyby scenario surfaces.
_MODE_A_HILL_CANDIDATES: tuple[str, ...] = (
    "mercury", "venus", "earth", "moon",
    "mars", "jupiter", "saturn", "uranus", "neptune",
)


def detect_hill_encounters(
    history: StateHistory,
    *,
    source: EphemerisSource | None = None,
) -> pd.DataFrame:
    """Return one row per encounter enter/exit found in `history`.

    Pairs every test particle with every massive body in the run
    (test particles are scenario-declared; massive bodies are everyone
    else *except* the Sun, which has no Hill sphere of its own here).
    For each pair, walks the per-sample distance and flags transitions
    across the body's Hill radius.

    `source` is required when `history.scenario.integrator.ephemeris_perturbers`
    is True (Mode A): in that case there are no scenario.bodies to pair
    against, so the detector reads major-body positions back from the
    supplied EphemerisSource at the same TDB instants the StateHistory
    samples were taken. When `source` is None and the scenario is Mode A,
    encounter detection is skipped and an empty event log is returned —
    not an error, since the runner's primary output is still the
    trajectory itself.
    """
    if history.df.empty:
        return _empty_events_df()

    test_particle_names = _test_particle_names(history)
    if not test_particle_names:
        return _empty_events_df()

    if history.scenario.integrator.ephemeris_perturbers:
        return _detect_mode_a(history, test_particle_names, source)

    return _detect_mode_b(history, test_particle_names)


def _detect_mode_b(
    history: StateHistory, test_particle_names: list[str],
) -> pd.DataFrame:
    """Mode B: massive bodies live in `history.df` alongside test particles.
    Pair them all using the existing pivot-based per-body trajectory."""
    massive_bodies = _massive_body_names(history)
    if not massive_bodies:
        return _empty_events_df()

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
        r_H = _hill_radius_km_from_pos(
            body_name, _xyz(body_name)[0], sun_pos[0] if sun_pos is not None else None,
        )
        if r_H is None:
            continue
        body_xyz = _xyz(body_name)
        for tp_name in test_particle_names:
            tp_xyz = _xyz(tp_name)
            d = np.linalg.norm(tp_xyz - body_xyz, axis=1)
            for ev in _crossings(d, r_H, body_name, tp_name, t_tdb):
                rows.append(ev.to_row())

    return _rows_to_df(rows)


def _detect_mode_a(
    history: StateHistory,
    test_particle_names: list[str],
    source: EphemerisSource | None,
) -> pd.DataFrame:
    """Mode A: scenario.bodies is empty. Pair test particles against
    skyfield-resolved major-body trajectories at the StateHistory's
    sample times. Skipped (returns empty event log) if `source` is
    None — the trajectory data in StateHistory is still authoritative;
    only the encounter side-output is missing."""
    if source is None:
        return _empty_events_df()

    pivot = (
        history.df.sort_values(["sample_idx", "body"])
        .pivot(index="sample_idx", columns="body", values=["x", "y", "z", "t_tdb"])
    )
    t_tdb = pivot["t_tdb"].iloc[:, 0].to_numpy(dtype=np.float64)
    sample_times = history.scenario.epoch + t_tdb * u.s

    # Bulk-query Sun first — its trajectory anchors every other body's
    # Hill radius via the sample-0 sun-relative distance.
    try:
        sun_xyz, _ = source.query_many("sun", sample_times)
    except UnknownBodyError:
        sun_xyz = None

    # Major-body positions: one vectorized skyfield call per body.
    # Bodies that the loaded kernels don't cover get skipped — that's
    # how an `EphemerisSource(de440s only)` setup gracefully degrades
    # vs. one that has the satellite kernels too.
    body_trajectories: dict[str, np.ndarray] = {}
    for body_name in _MODE_A_HILL_CANDIDATES:
        try:
            r_arr, _ = source.query_many(body_name, sample_times)
        except UnknownBodyError:
            continue
        body_trajectories[body_name] = r_arr

    if not body_trajectories:
        return _empty_events_df()

    # Test-particle trajectories pivot from the StateHistory.
    def _xyz(name: str) -> np.ndarray:
        return np.stack(
            [pivot[("x", name)], pivot[("y", name)], pivot[("z", name)]],
            axis=-1,
        )

    rows: list[dict[str, object]] = []
    for body_name, body_xyz in body_trajectories.items():
        r_H = _hill_radius_km_from_pos(
            body_name, body_xyz[0], sun_xyz[0] if sun_xyz is not None else None,
        )
        if r_H is None:
            continue
        for tp_name in test_particle_names:
            tp_xyz = _xyz(tp_name)
            d = np.linalg.norm(tp_xyz - body_xyz, axis=1)
            for ev in _crossings(d, r_H, body_name, tp_name, t_tdb):
                rows.append(ev.to_row())

    return _rows_to_df(rows)


def _rows_to_df(rows: list[dict[str, object]]) -> pd.DataFrame:
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


def _hill_radius_km_from_pos(
    body_name: str,
    body_pos_km: np.ndarray,
    sun_pos_km: np.ndarray | None,
) -> float | None:
    """r_H = a · (m / (3 M_sun))^(1/3) using `body_pos_km` (a single
    sample-0 position vector) and `sun_pos_km` (sample-0 Sun position
    or None for barycentric fallback) to derive `a`.

    Returns None when the body's mass isn't in BODY_CONSTANTS — test
    particles never have one, and unknown major bodies (e.g. a custom
    moon a user wired in via explicit IC without a constant) get
    skipped silently."""
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

    if sun_pos_km is not None:
        a_km = float(np.linalg.norm(body_pos_km - sun_pos_km))
    else:
        # No Sun in the scenario — fall back to barycentric distance.
        a_km = float(np.linalg.norm(body_pos_km))
    if a_km <= 0.0:
        return None
    return float(a_km * (m_kg / (3.0 * sun_const_mass)) ** (1.0 / 3.0))


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
        "dv_x_kms": "float64",
        "dv_y_kms": "float64",
        "dv_z_kms": "float64",
    }[col]
