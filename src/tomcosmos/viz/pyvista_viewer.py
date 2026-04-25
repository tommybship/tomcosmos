"""PyVista 3D viewer — desktop rendering for a StateHistory.

Reads a run's trajectory and shows the solar system in ICRF barycentric,
with orbit trails, body labels, and a time slider. The viewer is the
visual sanity check for M1: if orbits close in the expected periods and
planets sit where you'd expect them, the physics is working.

Body scaling (PLAN.md > "Body scaling"): default is **log-exaggerated**
(radii ×~235 so Earth renders at ~1% of 1 AU). True scale is available
but basically invisible; fixed-size markers are deferred.

Camera modes: top-down ecliptic (default) + free trackball (pyvista's
default interaction). Follow-body and rotating-frame cameras are M3+
concerns; the hooks in this class make them easy to add later.

The render context is only created in `show()` / `screenshot()`, so the
Viewer can be *constructed* headless (tests do this). All mesh and
actor work still needs a running VTK context, which is why viewer
snapshot tests are marked `viewer` and gated on CI.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import pyvista as pv
import vtk
from astropy import units as u

from tomcosmos.constants import resolve_body_constant
from tomcosmos.exceptions import UnknownBodyError
from tomcosmos.io.history import StateHistory

_AU_KM: float = float((1.0 * u.AU).to(u.km).value)
_EARTH_RADIUS_KM: float = 6371.0

# Log-scaling parameters. PLAN.md proposes `display = true_r * k` with k chosen
# so Earth is ~1% of 1 AU, but Sun (109x Earth) then renders at >1 AU and
# engulfs the scene. A power-law compression with exponent 0.3 keeps every
# body visible on the same zoom: Sun ~4x Earth, Jupiter ~2x Earth, Mercury
# ~0.75x. Earth's display radius is fixed by _LOG_EARTH_TARGET_AU.
_LOG_EARTH_TARGET_AU: float = 0.005
_LOG_SCALE_EXPONENT: float = 0.3

_DEFAULT_COLOR = "#CCCCCC"  # for bodies not in BODY_CONSTANTS

Scaling = Literal["true", "log", "marker"]


class Viewer:
    """Holds the scene graph for a StateHistory and exposes show()/screenshot().

    Construction builds per-body meshes, trail polylines, and the time
    slider widget, but doesn't open a window until `show()` runs. Tests
    pass `off_screen=True` to construct the Plotter in a headless mode
    suitable for snapshot comparison.
    """

    def __init__(
        self,
        history: StateHistory,
        *,
        scaling: Scaling = "log",
        follow: str | None = None,
        off_screen: bool = False,
    ) -> None:
        self._history = history
        self._scaling = scaling
        self._positions_au = _positions_by_body(history)
        self._body_names: tuple[str, ...] = tuple(self._positions_au.keys())
        if follow is not None and follow not in self._positions_au:
            raise ValueError(
                f"follow={follow!r} not in StateHistory bodies "
                f"({sorted(self._body_names)})"
            )
        self._follow = follow
        self._n_samples = history.n_samples
        self._current_sample = 0
        self._plotter = pv.Plotter(off_screen=off_screen, title="tomcosmos")
        self._body_actors: dict[str, pv.Actor] = {}
        self._label_polydata: pv.PolyData | None = None
        self._build_scene()

    # --- Public surface --------------------------------------------------

    @property
    def n_samples(self) -> int:
        return self._n_samples

    @property
    def body_names(self) -> tuple[str, ...]:
        return self._body_names

    def show(self) -> None:
        """Open the interactive window (blocks until closed)."""
        self._plotter.show()

    def screenshot(self, path: str) -> None:
        """Write a PNG at the current camera. Useful for snapshot tests."""
        self._plotter.screenshot(path)

    def set_sample(self, sample_idx: int) -> None:
        """Move every body to its position at `sample_idx`."""
        if not (0 <= sample_idx < self._n_samples):
            raise IndexError(
                f"sample_idx {sample_idx} out of range [0, {self._n_samples - 1}]"
                )
        for name, actor in self._body_actors.items():
            pos = self._positions_au[name][sample_idx]
            actor.position = (float(pos[0]), float(pos[1]), float(pos[2]))
        if self._label_polydata is not None:
            self._label_polydata.points = np.array(
                [self._positions_au[n][sample_idx] for n in self._body_names],
                dtype=np.float64,
            )
        # Follow-body camera: re-center the focal point on the followed body
        # each frame so it stays fixed in the viewport while everything else
        # moves around it. Without this, the camera stays at SSB and the
        # followed body slides out of frame.
        if self._follow is not None:
            target = self._positions_au[self._follow][sample_idx]
            cam = self._plotter.camera
            # Preserve the camera offset from the previous focal point
            # so the user's interactive rotation/zoom is retained.
            old_focal = np.array(cam.focal_point, dtype=np.float64)
            old_pos = np.array(cam.position, dtype=np.float64)
            offset = old_pos - old_focal
            new_focal = np.array(target, dtype=np.float64)
            cam.focal_point = (float(new_focal[0]), float(new_focal[1]), float(new_focal[2]))
            new_pos = new_focal + offset
            cam.position = (float(new_pos[0]), float(new_pos[1]), float(new_pos[2]))
        self._current_sample = sample_idx

    # --- Scene construction ---------------------------------------------

    def _build_scene(self) -> None:
        self._plotter.set_background("black")  # type: ignore[arg-type]
        self._add_trails()
        self._add_bodies()
        self._add_labels()
        self._set_top_down_camera()
        if self._n_samples > 1:
            self._add_time_slider()

    def _add_trails(self) -> None:
        """Full-history polyline per body, rendered once up front."""
        for name, pts in self._positions_au.items():
            if len(pts) < 2:
                continue
            color = _color_for(name)
            polyline = pv.MultipleLines(points=pts)
            self._plotter.add_mesh(
                polyline, color=color, line_width=1.5, opacity=0.6,
            )

    def _add_bodies(self) -> None:
        for name, pts in self._positions_au.items():
            radius_au = _display_radius_au(name, self._scaling)
            sphere = pv.Sphere(radius=radius_au, center=(0.0, 0.0, 0.0))
            actor = self._plotter.add_mesh(
                sphere, color=_color_for(name), smooth_shading=True,
            )
            actor.position = (float(pts[0, 0]), float(pts[0, 1]), float(pts[0, 2]))
            self._body_actors[name] = actor

    def _add_labels(self) -> None:
        """Body-name labels that track each body as the slider scrubs.

        We attach labels as a point-data string array and pass its
        *name* to `add_point_labels`. Pyvista's list-of-labels path
        deep-copies the input polydata, which would freeze the labels;
        the named-array path keeps the dataset by reference, so
        `set_sample` can mutate `_label_polydata.points` and the label
        hierarchy follows on the next render. Anchor dots are
        suppressed — the body sphere is the only marker.
        """
        points = np.array(
            [self._positions_au[n][0] for n in self._body_names],
            dtype=np.float64,
        )
        self._label_polydata = pv.PolyData(points)
        labels_arr = vtk.vtkStringArray()
        labels_arr.SetName("labels")
        for name in self._body_names:
            labels_arr.InsertNextValue(name)
        self._label_polydata.GetPointData().AddArray(labels_arr)
        self._plotter.add_point_labels(
            self._label_polydata, "labels",
            font_size=12, text_color="white",
            shape=None, always_visible=True, show_points=False,
        )

    def _set_top_down_camera(self) -> None:
        """Frame the scene from +Z looking down at the ecliptic, parallel
        projection scaled to the outermost trajectory.

        Orthographic (parallel) projection is the right default for a
        solar-system layout: perspective makes Neptune look smaller than
        Jupiter at the same display scale, and it clips the outer
        planets when the inner system is centered.

        In follow-body mode, the focal point centers on the followed
        body's t=0 position and the parallel scale is chosen from the
        spread of *other* bodies in the followed body's neighborhood,
        not from absolute distance to SSB. This makes a Galilean
        moon's orbit around Jupiter visible at appropriate zoom.
        """
        if self._follow is not None:
            follow_t0 = self._positions_au[self._follow][0]
            # Per-body MAX distance from the followed body over the run. A
            # good camera scale is "the cohort of bodies that orbit the
            # followed body" — Earth-Moon for `follow=earth`, Galileans for
            # `follow=jupiter`. We want to exclude distant primaries like
            # the Sun when following a planet.
            n_probe = min(self._n_samples, 32)
            sample_idxs = np.linspace(0, self._n_samples - 1, n_probe, dtype=int)
            follow_pts = self._positions_au[self._follow]
            per_body_max: list[float] = []
            for name, pts in self._positions_au.items():
                if name == self._follow:
                    continue
                d = float(max(np.linalg.norm(pts[i] - follow_pts[i]) for i in sample_idxs))
                per_body_max.append(d)
            if per_body_max:
                # The closest body sets the floor. Include bodies up to 5x
                # that floor — i.e., the followed body's tight neighborhood.
                # Anything farther (typically the central star or distant
                # planet) gets cropped from the framing, even though it's
                # still rendered in the scene.
                closest = min(per_body_max)
                cohort = [d for d in per_body_max if d <= closest * 5.0]
                spread = max(cohort)
            else:
                spread = 0.05  # fallback when only the followed body is in scene
            self._plotter.camera_position = [
                (float(follow_t0[0]), float(follow_t0[1]), float(follow_t0[2]) + spread * 3.0),
                (float(follow_t0[0]), float(follow_t0[1]), float(follow_t0[2])),
                (0.0, 1.0, 0.0),
            ]
            self._plotter.enable_parallel_projection()  # type: ignore[call-arg]
            self._plotter.camera.parallel_scale = spread * 1.5
            return

        max_r = max(
            float(np.linalg.norm(pts, axis=1).max())
            for pts in self._positions_au.values()
            if len(pts) > 0
        )
        if max_r == 0:
            max_r = 1.0
        self._plotter.camera_position = [
            (0.0, 0.0, max_r * 3.0),   # position; distance doesn't matter in parallel
            (0.0, 0.0, 0.0),           # focal point
            (0.0, 1.0, 0.0),           # view up
        ]
        self._plotter.enable_parallel_projection()  # type: ignore[call-arg]
        self._plotter.camera.parallel_scale = max_r * 1.1

    def _add_time_slider(self) -> None:
        def _on_slider(value: float) -> None:
            self.set_sample(int(round(value)))

        self._plotter.add_slider_widget(
            _on_slider,
            rng=(0, self._n_samples - 1),
            value=0,
            title="sample",
            pointa=(0.1, 0.05), pointb=(0.9, 0.05),
            style="modern",
            interaction_event="always",
        )


def _positions_by_body(history: StateHistory) -> dict[str, np.ndarray]:
    """Per-body (n_samples, 3) position array in AU, ordered by sample_idx."""
    out: dict[str, np.ndarray] = {}
    df = history.df.sort_values("sample_idx")
    for name in history.body_names:
        sub = df[df["body"] == name]
        pts_km = sub[["x", "y", "z"]].to_numpy(dtype=np.float64)
        out[name] = pts_km / _AU_KM
    return out


def _display_radius_au(body_name: str, scaling: Scaling) -> float:
    """Per-body display radius in AU according to the chosen scaling mode."""
    try:
        const = resolve_body_constant(body_name)
        radius_km = const.radius_km
    except UnknownBodyError:
        radius_km = _EARTH_RADIUS_KM  # fall back to Earth-ish if unknown

    if scaling == "true":
        return radius_km / _AU_KM
    if scaling == "log":
        ratio = radius_km / _EARTH_RADIUS_KM
        return float(_LOG_EARTH_TARGET_AU * (ratio ** _LOG_SCALE_EXPONENT))
    if scaling == "marker":
        # Fixed small AU-sized marker — good enough until billboarded
        # markers arrive in a later iteration.
        return 0.01
    raise ValueError(f"unknown scaling mode: {scaling!r}")


def _color_for(body_name: str) -> str:
    try:
        return resolve_body_constant(body_name).color_hex
    except UnknownBodyError:
        return _DEFAULT_COLOR


__all__ = ["Viewer", "Scaling"]
