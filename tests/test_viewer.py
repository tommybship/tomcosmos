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
from tomcosmos.state.ephemeris import SkyfieldSource

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


class _NoEphemerisNeeded(SkyfieldSource):  # type: ignore[misc]
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
