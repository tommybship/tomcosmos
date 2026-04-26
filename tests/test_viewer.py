"""Viewer tests — plumbing only.

Full snapshot tests (rendering comparisons against committed PNGs) are
the PLAN.md `pytest.mark.viewer` tier; these just exercise construction
and state-updates so we catch wiring regressions without a display.
Marked `viewer` so CI skips them unless xvfb is available.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tomcosmos import Scenario, run
from tomcosmos.state.ephemeris import EphemerisSource

pytestmark = pytest.mark.viewer

AU_KM = 1.495978707e8


def _explicit_scenario() -> Scenario:
    return Scenario.model_validate(
        {
            "schema_version": 1,
            "name": "viewer-test",
            "epoch": "2026-01-01T00:00:00 TDB",
            "duration": "30 day",
            "integrator": {"name": "whfast", "timestep": "1 day"},
            "output": {"format": "parquet", "cadence": "5 day"},
            "bodies": [
                {"name": "sun", "mass_kg": 1.989e30, "radius_km": 695700.0,
                 "ic": {"source": "explicit", "r": [0, 0, 0], "v": [0, 0, 0]}},
                {"name": "earth", "mass_kg": 5.9724e24, "radius_km": 6371.0,
                 "ic": {"source": "explicit",
                        "r": [AU_KM, 0, 0], "v": [0, 29.7847, 0]}},
            ],
        }
    )


class _NoEphemerisNeeded(EphemerisSource):  # type: ignore[misc]
    def __init__(self) -> None: pass
    def query(self, body, epoch): raise AssertionError  # type: ignore[no-untyped-def]
    def available_bodies(self): return ()  # type: ignore[override]
    def time_range(self):  # type: ignore[override]
        from astropy.time import Time
        return Time("1900-01-01", scale="tdb"), Time("2100-01-01", scale="tdb")


def _fresh_history():  # type: ignore[no-untyped-def]
    return run(_explicit_scenario(), source=_NoEphemerisNeeded())


# --- Construction ------------------------------------------------------------


def test_viewer_constructs_off_screen() -> None:
    from tomcosmos.viz.pyvista_viewer import Viewer
    history = _fresh_history()
    viewer = Viewer(history, off_screen=True)
    assert viewer.n_samples == history.n_samples
    assert set(viewer.body_names) == {"sun", "earth"}


def test_viewer_registers_body_actors() -> None:
    from tomcosmos.viz.pyvista_viewer import Viewer
    history = _fresh_history()
    viewer = Viewer(history, off_screen=True)
    # Internal dict is private but exposed for plumbing tests.
    assert set(viewer._body_actors.keys()) == {"sun", "earth"}  # noqa: SLF001


def test_set_sample_updates_positions() -> None:
    from tomcosmos.viz.pyvista_viewer import Viewer
    history = _fresh_history()
    viewer = Viewer(history, off_screen=True)
    viewer.set_sample(0)
    p0 = np.array(viewer._body_actors["earth"].GetPosition())  # noqa: SLF001
    viewer.set_sample(history.n_samples - 1)
    p1 = np.array(viewer._body_actors["earth"].GetPosition())  # noqa: SLF001
    assert not np.allclose(p0, p1), "Earth should move across samples"


def test_set_sample_updates_label_polydata() -> None:
    """Labels track the bodies, not the t=0 anchor.

    Guards against pyvista's add_point_labels deep-copy gotcha: when
    `labels` is passed as a Python list (not a point-data array name),
    the input polydata is copied internally and our handle becomes
    stale, freezing labels at sample 0.
    """
    from tomcosmos.viz.pyvista_viewer import Viewer
    history = _fresh_history()
    viewer = Viewer(history, off_screen=True)
    assert viewer._label_polydata is not None  # noqa: SLF001
    earth_idx = viewer.body_names.index("earth")

    viewer.set_sample(0)
    p0 = viewer._label_polydata.points[earth_idx].copy()  # noqa: SLF001
    viewer.set_sample(history.n_samples - 1)
    p1 = viewer._label_polydata.points[earth_idx]  # noqa: SLF001

    assert not np.allclose(p0, p1), "Earth label should move with the body"
    assert np.allclose(p1, viewer._positions_au["earth"][-1])  # noqa: SLF001


def test_set_sample_bounds_check() -> None:
    from tomcosmos.viz.pyvista_viewer import Viewer
    history = _fresh_history()
    viewer = Viewer(history, off_screen=True)
    with pytest.raises(IndexError):
        viewer.set_sample(-1)
    with pytest.raises(IndexError):
        viewer.set_sample(history.n_samples)


def test_scaling_mode_validated() -> None:
    from tomcosmos.viz.pyvista_viewer import Viewer
    history = _fresh_history()
    with pytest.raises(ValueError, match="unknown scaling"):
        Viewer(history, scaling="cube", off_screen=True)  # type: ignore[arg-type]


# --- Smoke rendering to a PNG ------------------------------------------------


def test_screenshot_writes_png(tmp_path: Path) -> None:
    """End-to-end: off-screen render + PNG write. Validates the full
    render pipeline without eyeballing pixels."""
    from tomcosmos.viz.pyvista_viewer import Viewer
    history = _fresh_history()
    viewer = Viewer(history, off_screen=True)
    out = tmp_path / "screenshot.png"
    viewer.screenshot(str(out))
    assert out.exists()
    assert out.stat().st_size > 1000  # non-trivial PNG


# --- Follow-body camera (M2c) -----------------------------------------------


def test_follow_body_constructs() -> None:
    from tomcosmos.viz.pyvista_viewer import Viewer
    history = _fresh_history()
    viewer = Viewer(history, follow="earth", off_screen=True)
    assert viewer._follow == "earth"  # noqa: SLF001


def test_follow_unknown_body_rejected() -> None:
    from tomcosmos.viz.pyvista_viewer import Viewer
    history = _fresh_history()
    with pytest.raises(ValueError, match="not in StateHistory bodies"):
        Viewer(history, follow="ganymede", off_screen=True)


def test_follow_camera_focal_point_tracks_body() -> None:
    """As the slider scrubs, the camera's focal point should follow the
    target body — that's what 'follow' does."""
    from tomcosmos.viz.pyvista_viewer import Viewer
    history = _fresh_history()
    viewer = Viewer(history, follow="earth", off_screen=True)

    viewer.set_sample(0)
    focal_0 = np.array(viewer._plotter.camera.focal_point)  # noqa: SLF001
    earth_0 = viewer._positions_au["earth"][0]  # noqa: SLF001
    assert np.allclose(focal_0, earth_0, atol=1e-9)

    last_idx = history.n_samples - 1
    viewer.set_sample(last_idx)
    focal_last = np.array(viewer._plotter.camera.focal_point)  # noqa: SLF001
    earth_last = viewer._positions_au["earth"][last_idx]  # noqa: SLF001
    assert np.allclose(focal_last, earth_last, atol=1e-9)
    # The focal point actually moved (not stuck at origin like default mode).
    assert not np.allclose(focal_0, focal_last)


def _bulk_test_particle_scenario(n: int) -> Scenario:
    """Sun plus `n` synthetic test particles on circular orbits — used
    for the bulk-cohort viewer path test."""
    import math

    rng = np.random.default_rng(0)
    bodies = [{
        "name": "sun", "mass_kg": 1.989e30, "radius_km": 695700.0,
        "ic": {"source": "explicit", "r": [0, 0, 0], "v": [0, 0, 0]},
    }]
    test_particles = []
    for i in range(n):
        a_au = 1.5 + 0.005 * i
        a_km = a_au * AU_KM
        ph = float(rng.uniform(0, 2 * math.pi))
        speed = 29.7847 / math.sqrt(a_au)
        test_particles.append({
            "name": f"p{i:04d}",
            "ic": {
                "type": "explicit", "frame": "icrf_barycentric",
                "r": [a_km * math.cos(ph), a_km * math.sin(ph), 0.0],
                "v": [-speed * math.sin(ph), speed * math.cos(ph), 0.0],
            },
        })
    return Scenario.model_validate({
        "schema_version": 1, "name": "bulk-test",
        "epoch": "2026-01-01T00:00:00 TDB", "duration": "365 day",
        "integrator": {"name": "whfast", "timestep": "5 day"},
        "output": {"format": "parquet", "cadence": "30 day"},
        "bodies": bodies, "test_particles": test_particles,
    })


def test_viewer_bulk_cohort_activates_above_threshold() -> None:
    """When a scenario declares more test particles than the bulk
    threshold, the viewer renders them as a single PolyData cloud
    instead of N sphere actors. Massive bodies (the Sun here) keep
    the per-body actor treatment regardless."""
    from tomcosmos.viz.pyvista_viewer import Viewer

    history = run(_bulk_test_particle_scenario(50), source=_NoEphemerisNeeded())
    viewer = Viewer(history, off_screen=True)
    # Sun stays as an actor; 50 test particles go to the bulk cohort.
    assert "sun" in viewer._body_actors  # noqa: SLF001
    assert len(viewer._bulk_cohort) == 50  # noqa: SLF001
    assert viewer._bulk_polydata is not None  # noqa: SLF001
    # The bulk cohort has no per-body actors — that's the whole point.
    for name in viewer._bulk_cohort:  # noqa: SLF001
        assert name not in viewer._body_actors  # noqa: SLF001


def test_viewer_bulk_cohort_set_sample_updates_polydata() -> None:
    """`set_sample` mutates the bulk PolyData's points in-place; after
    a scrub, the points coordinate at the rendered sample matches the
    history's positions for the bulk cohort. Trip wire for the
    points-update path silently no-op'ing in a future refactor."""
    from tomcosmos.viz.pyvista_viewer import Viewer

    n_tp = 50
    history = run(_bulk_test_particle_scenario(n_tp), source=_NoEphemerisNeeded())
    viewer = Viewer(history, off_screen=True)

    target_idx = history.n_samples // 2
    viewer.set_sample(target_idx)

    rendered = np.asarray(viewer._bulk_polydata.points)  # noqa: SLF001
    assert rendered.shape == (n_tp, 3)
    expected = viewer._bulk_positions_au[target_idx]  # noqa: SLF001
    np.testing.assert_allclose(rendered, expected, atol=1e-12)


def test_viewer_below_threshold_uses_per_body_actors() -> None:
    """A 2-body scenario stays on the per-body actor path — the bulk
    cohort kicks in only when it earns its keep."""
    from tomcosmos.viz.pyvista_viewer import Viewer

    viewer = Viewer(_fresh_history(), off_screen=True)
    assert viewer._bulk_cohort == ()  # noqa: SLF001
    assert viewer._bulk_polydata is None  # noqa: SLF001
    assert {"sun", "earth"} <= set(viewer._body_actors)  # noqa: SLF001


def test_follow_camera_does_not_apply_in_default_mode() -> None:
    """Without follow, the focal point stays at the SSB origin throughout."""
    from tomcosmos.viz.pyvista_viewer import Viewer
    history = _fresh_history()
    viewer = Viewer(history, off_screen=True)
    viewer.set_sample(0)
    focal_0 = np.array(viewer._plotter.camera.focal_point)  # noqa: SLF001
    viewer.set_sample(history.n_samples - 1)
    focal_last = np.array(viewer._plotter.camera.focal_point)  # noqa: SLF001
    assert np.allclose(focal_0, focal_last)
