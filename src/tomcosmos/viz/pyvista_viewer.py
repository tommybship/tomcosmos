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

import re
from typing import Literal

import numpy as np
import pyvista as pv
import vtk
from astropy import units as u
from astropy.time import TimeDelta

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

# J2000.0 epoch in TDB Julian Date — anchor for the IAU rotation
# model's W(t) = W₀ + W' × (Δd) where Δd = (t.tdb.jd - J2000_TDB_JD).
_J2000_TDB_JD: float = 2451545.0

# Display radius (in km) for bodies absent from BODY_CONSTANTS. Earth's
# radius was the original fallback, but that's catastrophic at close-up
# scales: a test particle 38,000 km from Earth gets rendered with the
# same display radius as Earth itself, engulfing the camera. 1 km is
# asteroid-scale and disappears at solar-system zoom — visible only when
# the user zooms in significantly, which is what they're doing if they
# care about an unnamed test particle.
_UNKNOWN_RADIUS_KM: float = 1.0

# When a scenario declares more test particles than this, the viewer
# renders them as a single point-cloud `pv.PolyData` instead of one
# `pv.Sphere` actor per particle. PyVista's per-actor rendering pipeline
# breaks down around a few hundred actors; the points-cloud path scales
# to tens of thousands. Massive bodies (Sun, planets, moons) keep the
# per-body actor treatment regardless — there are never enough of them
# to matter, and the extra detail (sized spheres, labels, trails) is
# the whole point of having them on screen.
_BULK_COHORT_THRESHOLD: int = 20
_BULK_POINT_COLOR_HEX: str = "#FFFFFF"

Scaling = Literal["true", "log", "marker", "auto"]


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
        rotating: tuple[str, str] | None = None,
        textures: bool = True,
        off_screen: bool = False,
    ) -> None:
        self._history = history
        self._scaling = scaling
        # Texture support: when True, bodies in the textures registry
        # (Earth ships offline; other planets fetch on first call) are
        # rendered with their UV-mapped meshes. The texture only becomes
        # visible at high zoom — at default solar-system framing,
        # planets are sub-pixel and the texture is invisible regardless.
        # Off-screen renders skip textures entirely: the snapshot tests
        # don't need them, and the network fetches would slow CI.
        self._use_textures = textures and not off_screen
        self._positions_au = _positions_by_body(history)
        self._body_names: tuple[str, ...] = tuple(self._positions_au.keys())
        # Rotating-frame mode: pre-rotate every sample of every body into
        # the (primary, secondary) corotating frame, then operate on the
        # rotated arrays. Trails, body actors, labels, and `set_sample`
        # all use _positions_au, so swapping it here is enough — the rest
        # of the pipeline doesn't need to know which frame it's in.
        if rotating is not None:
            primary, secondary = rotating
            for required in (primary, secondary):
                if required not in self._positions_au:
                    raise ValueError(
                        f"rotating={rotating!r}: {required!r} not in StateHistory "
                        f"bodies ({sorted(self._body_names)})"
                    )
            from tomcosmos.analysis.rotating_frame import (
                rotate_history_to_corotating,
            )
            rotated_km = rotate_history_to_corotating(history, primary, secondary)
            self._positions_au = {n: pts / _AU_KM for n, pts in rotated_km.items()}
        self._rotating = rotating
        if follow is not None and follow not in self._positions_au:
            raise ValueError(
                f"follow={follow!r} not in StateHistory bodies "
                f"({sorted(self._body_names)})"
            )
        if rotating is not None and follow is not None:
            raise ValueError(
                "follow and rotating cannot be combined; rotating already keeps "
                "the secondary fixed in the viewport"
            )
        self._follow = follow
        self._n_samples = history.n_samples
        self._current_sample = 0
        self._plotter = pv.Plotter(off_screen=off_screen, title="tomcosmos")
        self._body_actors: dict[str, pv.Actor] = {}
        self._label_actors: dict[str, vtk.vtkBillboardTextActor3D] = {}

        # Decide which names render as the bulk point cloud vs. as
        # individual sphere actors. Only test particles are eligible —
        # massive bodies always get the full treatment.
        tp_names = [p.name for p in history.scenario.test_particles]
        if len(tp_names) > _BULK_COHORT_THRESHOLD:
            self._bulk_cohort: tuple[str, ...] = tuple(tp_names)
        else:
            self._bulk_cohort = ()
        self._bulk_cohort_set = frozenset(self._bulk_cohort)
        # (n_samples, n_bulk, 3) array of AU positions, ready for slice-by-sample
        # writes into the bulk PolyData. Built once up front because slicing
        # this is much faster than the per-name dict lookups in set_sample.
        if self._bulk_cohort:
            self._bulk_positions_au: np.ndarray = np.stack(
                [self._positions_au[n] for n in self._bulk_cohort], axis=1,
            )
        else:
            self._bulk_positions_au = np.empty((self._n_samples, 0, 3), dtype=np.float64)
        self._bulk_polydata: pv.PolyData | None = None
        self._time_overlay_actor: object | None = None
        self._scaling_K: float = self._compute_auto_scaling_k()
        # (n_samples, 3, 3) IAU rotation matrices per body that has
        # rotation data. Built once in _build_time_axis (which already
        # computes per-sample J2000-offset days for the slider). Empty
        # for bodies without IAU data — they don't spin.
        self._rotation_matrices: dict[str, np.ndarray] = {}
        self._build_time_axis()
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
            r_matrix = self._rotation_matrices.get(name)
            if r_matrix is None:
                actor.position = (float(pos[0]), float(pos[1]), float(pos[2]))
            else:
                # vtkProp3D's PostMultiply transform stack applies the
                # UserMatrix LAST, so it rotates the actor's translated
                # position too: p_world = UserMatrix * (mesh_p + pos)
                # = R*mesh_p + R*pos. To avoid the unwanted R*pos, bake
                # the position into the matrix's translation column and
                # zero actor.position so VTK has nothing to rotate.
                m = np.eye(4)
                m[:3, :3] = r_matrix[sample_idx]
                m[:3, 3] = pos
                actor.position = (0.0, 0.0, 0.0)
                actor.user_matrix = m
        for name, label_actor in self._label_actors.items():
            pos = self._positions_au[name][sample_idx]
            label_actor.SetPosition(float(pos[0]), float(pos[1]), float(pos[2]))
        if self._bulk_polydata is not None:
            # One contiguous slice write — VTK picks up the change on
            # the next render. Cost is O(n_bulk) per frame regardless
            # of cohort size.
            self._bulk_polydata.points = self._bulk_positions_au[sample_idx]
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
        if self._time_overlay_actor is not None:
            # CornerAnnotation: SetText(corner_idx, text). pyvista's
            # position="upper_left" maps to VTK corner index 2.
            self._time_overlay_actor.SetText(2, self._abs_time_strs[sample_idx])
        self._current_sample = sample_idx

    # --- Scene construction ---------------------------------------------

    def _build_scene(self) -> None:
        self._plotter.set_background("black")
        self._add_trails()
        self._add_bodies()
        self._add_bulk_points()
        self._add_labels()
        self._set_top_down_camera()
        self._add_time_overlay()
        if self._n_samples > 1:
            self._add_time_slider()
        # Apply sample-0 transforms (positions + IAU rotation matrices)
        # uniformly through set_sample, so initial render isn't a
        # special-cased path that diverges from scrubbed frames.
        if self._n_samples > 0:
            self.set_sample(0)

    def _build_time_axis(self) -> None:
        """Pre-compute per-sample time axis: seconds-from-epoch, the
        slider's chosen unit, and absolute timestamp strings.

        Done once in __init__ so the slider callback and overlay update
        don't pay astropy cost per scrub. For a 36k-sample run, this is
        ~few-MB of strings — negligible.
        """
        df = self._history.df.sort_values("sample_idx")
        per_sample = df.drop_duplicates("sample_idx", keep="first")
        t_seconds = per_sample["t_tdb"].to_numpy(dtype=np.float64)

        total_days = float(t_seconds[-1]) / 86400.0 if t_seconds.size else 0.0
        if total_days <= 30.0:
            unit_name, unit_seconds = "hour", 3600.0
        elif total_days <= 5.0 * 365.25:
            unit_name, unit_seconds = "day", 86400.0
        else:
            unit_name, unit_seconds = "year", 365.25 * 86400.0
        self._slider_unit_name = unit_name
        self._t_per_sample_in_unit = t_seconds / unit_seconds

        epoch = self._history.scenario.epoch
        times = epoch + TimeDelta(t_seconds, format="sec")
        # times.tdb.isot returns an ndarray of strings; coerce to a list
        # for cheap indexed access in set_sample.
        self._abs_time_strs: list[str] = [
            f"T = {s} TDB" for s in np.atleast_1d(times.tdb.isot).tolist()
        ]
        # Per-sample IAU rotation matrices for every body that has
        # rotational elements registered. Pre-computed once so set_sample
        # is just a dict lookup + matrix copy into the user-matrix slot.
        days_past_j2000 = np.atleast_1d(times.tdb.jd - _J2000_TDB_JD).astype(np.float64)
        for name in self._body_names:
            try:
                const = resolve_body_constant(name)
            except UnknownBodyError:
                continue
            if (const.pole_ra_deg is None or const.pole_dec_deg is None
                or const.prime_meridian_at_j2000_deg is None
                or const.rotation_rate_deg_per_day is None):
                continue
            self._rotation_matrices[name] = _iau_rotation_matrices(
                alpha0_deg=const.pole_ra_deg,
                delta0_deg=const.pole_dec_deg,
                w0_deg=const.prime_meridian_at_j2000_deg,
                w_rate_deg_per_day=const.rotation_rate_deg_per_day,
                days_past_j2000=days_past_j2000,
            )

    def _add_time_overlay(self) -> None:
        """Top-left text actor showing the current sample's absolute
        timestamp in TDB. Mutated in place by `set_sample`."""
        if not self._abs_time_strs:
            return
        self._time_overlay_actor = self._plotter.add_text(
            self._abs_time_strs[0],
            position="upper_left",
            font_size=10,
            color="white",
            name="time_overlay",
        )

    def _add_trails(self) -> None:
        """Full-history polyline per body, rendered once up front. Skipped
        for the bulk cohort — drawing 1,000 polylines drowns the scene
        and tanks the framerate."""
        for name, pts in self._positions_au.items():
            if name in self._bulk_cohort_set:
                continue
            if len(pts) < 2:
                continue
            color = _color_for(name)
            polyline = pv.MultipleLines(points=pts)
            self._plotter.add_mesh(
                polyline, color=color, line_width=1.5, opacity=0.6,
            )

    def _compute_auto_scaling_k(self) -> float:
        """For scaling='auto': pick a uniform exaggeration K so each body
        renders large enough to see without engulfing nearby orbits.

        The rule: K = 0.15 * (smallest pairwise min separation in the
        cohort) / (largest true radius in the cohort), clamped to >= 1.0
        so we never shrink below physical scale.

        In follow mode the cohort is the followed body plus its tight
        neighborhood (matches `_set_top_down_camera`'s cohort detection),
        so the central star — light-years bigger than a moon and an
        AU+ away — doesn't constrain the framing of an Earth-Moon view.

        In non-follow mode there's no natural cohort so we return 1.0,
        and `_display_radius_au` falls back to log scaling — which is
        what made solar-system views work before this scaling existed.
        """
        if self._scaling != "auto":
            return 1.0
        if self._follow is None:
            # No cohort to compute K from — signal log fallback via 0.0.
            # `_display_radius_au` treats K==0 as the only "use log" case,
            # so K==1 from a tight scenario still means true scale.
            return 0.0
        n_probe = min(self._n_samples, 32)
        if n_probe < 1:
            return 1.0
        sample_idxs = np.linspace(0, self._n_samples - 1, n_probe, dtype=int)
        follow_pts = self._positions_au[self._follow]
        per_body_max: dict[str, float] = {}
        for name, pts in self._positions_au.items():
            if name == self._follow:
                continue
            d = float(max(
                np.linalg.norm(pts[i] - follow_pts[i]) for i in sample_idxs
            ))
            per_body_max[name] = d
        if not per_body_max:
            return 1.0
        closest = min(per_body_max.values())
        cohort = [n for n, d in per_body_max.items() if d <= closest * 5.0]
        cohort_with_focal = [self._follow, *cohort]
        if len(cohort_with_focal) < 2:
            return 1.0
        # Smallest separation across the run, considering every pair
        # where at least one side is in the cohort. A body that's not in
        # the cohort overall but flies through it briefly (like Apophis
        # threading past Earth) still constrains K — otherwise the
        # rendered cohort body's display sphere can engulf the visitor
        # at closest approach.
        cohort_set = set(cohort_with_focal)
        all_names = list(self._positions_au.keys())
        min_sep_au = float("inf")
        for i, name_i in enumerate(all_names):
            for name_j in all_names[i + 1:]:
                if name_i not in cohort_set and name_j not in cohort_set:
                    continue
                pts_i = self._positions_au[name_i]
                pts_j = self._positions_au[name_j]
                d = float(np.linalg.norm(pts_i - pts_j, axis=1).min())
                if d < min_sep_au:
                    min_sep_au = d
        if not np.isfinite(min_sep_au) or min_sep_au <= 0:
            return 1.0
        # Largest true radius in the cohort.
        max_r_km = 0.0
        for name in cohort_with_focal:
            try:
                r_km = resolve_body_constant(name).radius_km
            except UnknownBodyError:
                stripped = re.sub(r"[^a-z0-9]+$", "", name.lower())
                if stripped and stripped != name.lower():
                    try:
                        r_km = resolve_body_constant(stripped).radius_km
                    except UnknownBodyError:
                        r_km = _UNKNOWN_RADIUS_KM
                else:
                    r_km = _UNKNOWN_RADIUS_KM
            if r_km > max_r_km:
                max_r_km = r_km
        if max_r_km <= 0:
            return 1.0
        k = 0.15 * (min_sep_au * _AU_KM) / max_r_km
        return max(k, 1.0)

    def _add_bodies(self) -> None:
        for name, pts in self._positions_au.items():
            if name in self._bulk_cohort_set:
                continue
            radius_au = _display_radius_au(name, self._scaling, self._scaling_K)
            mesh, texture = self._load_textured_or_sphere(name, radius_au)
            if texture is not None:
                actor = self._plotter.add_mesh(
                    mesh, texture=texture, smooth_shading=True,
                )
            else:
                actor = self._plotter.add_mesh(
                    mesh, color=_color_for(name), smooth_shading=True,
                )
            actor.position = (float(pts[0, 0]), float(pts[0, 1]), float(pts[0, 2]))
            self._body_actors[name] = actor

    def _load_textured_or_sphere(
        self, name: str, radius_au: float,
    ) -> tuple[pv.PolyData, object | None]:
        """Return `(mesh, texture)` for body `name`. Texture is None when
        textures are disabled, when the body has no entry in the registry,
        or when the registry's loader fails (e.g. a download error for a
        non-Earth planet on a fresh machine without network). The
        fallback path always succeeds — it's just `pv.Sphere`."""
        if self._use_textures:
            from tomcosmos.viz.textures import load_for_body
            try:
                pair = load_for_body(name, radius_au)
            except Exception:  # noqa: BLE001 — pyvista downloads can raise widely
                pair = None
            if pair is not None:
                return pair
        return pv.Sphere(radius=radius_au, center=(0.0, 0.0, 0.0)), None

    def _add_bulk_points(self) -> None:
        """Single PolyData containing one vertex per bulk-cohort particle,
        rendered as point sprites. `set_sample` mutates `points` in-place;
        VTK picks up the change on the next render. O(n_bulk) per frame
        regardless of cohort size — this is what makes 1,000-asteroid
        scenarios interactive."""
        if not self._bulk_cohort:
            return
        # Initial positions are sample 0; set_sample will rewrite them.
        self._bulk_polydata = pv.PolyData(self._bulk_positions_au[0])
        self._plotter.add_mesh(
            self._bulk_polydata,
            color=_BULK_POINT_COLOR_HEX,
            render_points_as_spheres=True,
            point_size=4,
            opacity=0.85,
        )

    def _add_labels(self) -> None:
        """Body-name labels that track each body as the slider scrubs.

        Each labelled body gets its own `vtkBillboardTextActor3D` —
        a 3D-anchored text actor that always faces the camera and
        renders at a fixed pixel size regardless of zoom. We chose
        billboard actors over `add_point_labels` (which uses
        `vtkLabelHierarchy`) because the hierarchy builds an octree
        from the polydata's bounding box at construction time and
        then culls points whose later positions fall outside it —
        producing the symptom of labels visible only in the time band
        where bodies are near their initial positions, and disappearing
        before / after as orbits carry them away.

        Bulk-cohort particles are skipped: 1,000 floating name labels
        is unreadable, and the rendering cost would defeat the whole
        point of the bulk path.
        """
        labelled = [n for n in self._body_names if n not in self._bulk_cohort_set]
        if not labelled:
            return
        renderer = self._plotter.renderer
        for name in labelled:
            actor = vtk.vtkBillboardTextActor3D()
            actor.SetInput(name)
            tprop = actor.GetTextProperty()
            tprop.SetColor(1.0, 1.0, 1.0)
            tprop.SetFontSize(12)
            pos = self._positions_au[name][0]
            actor.SetPosition(float(pos[0]), float(pos[1]), float(pos[2]))
            renderer.AddActor(actor)
            self._label_actors[name] = actor

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
            self._plotter.enable_parallel_projection()
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
        self._plotter.enable_parallel_projection()
        self._plotter.camera.parallel_scale = max_r * 1.1

    def _add_time_slider(self) -> None:
        t_axis = self._t_per_sample_in_unit
        t_max = float(t_axis[-1])
        n = self._n_samples

        def _on_slider(value: float) -> None:
            # Map the slider's continuous time value back to the closest
            # discrete sample. searchsorted is O(log n) — fine for the
            # interactive scrub, regardless of run length.
            i = int(np.searchsorted(t_axis, value))
            if i >= n:
                i = n - 1
            elif i > 0 and abs(t_axis[i] - value) > abs(t_axis[i - 1] - value):
                i = i - 1
            self.set_sample(i)

        self._plotter.add_slider_widget(
            _on_slider,
            rng=(0.0, t_max),
            value=0.0,
            title=f"{self._slider_unit_name} from epoch",
            # y=0.10 (not 0.05) so the title rendered below the track has
            # room before the bottom of the window — at 0.05 the title's
            # descenders clip.
            pointa=(0.1, 0.10), pointb=(0.9, 0.10),
            style="modern",
            interaction_event="always",
            fmt="%.2f",
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


def _display_radius_au(
    body_name: str, scaling: Scaling, k_auto: float = 1.0,
) -> float:
    """Per-body display radius in AU according to the chosen scaling mode.

    `k_auto` is the exaggeration factor produced by
    `Viewer._compute_auto_scaling_k()`. Only consulted when
    `scaling=='auto'`; ignored for the other modes. A value of 1.0
    means the auto path falls back to log behavior (e.g. when there's
    no follow target to define a cohort).
    """
    try:
        const = resolve_body_constant(body_name)
        radius_km = const.radius_km
    except UnknownBodyError:
        # Suffix-aware fallback: scripts/overlay_runs.py renames overlay
        # bodies by appending a marker (default '*') so colors fall back
        # to grey via BODY_CONSTANTS miss. Sizes shouldn't fall back the
        # same way — strip a trailing non-alphanumeric suffix and retry
        # so e.g. 'apophis*' picks up Apophis's real 0.17 km radius.
        stripped = re.sub(r"[^a-z0-9]+$", "", body_name.lower())
        if stripped and stripped != body_name.lower():
            try:
                radius_km = resolve_body_constant(stripped).radius_km
            except UnknownBodyError:
                radius_km = _UNKNOWN_RADIUS_KM
        else:
            radius_km = _UNKNOWN_RADIUS_KM

    if scaling == "true":
        return radius_km / _AU_KM
    if scaling == "log":
        ratio = radius_km / _EARTH_RADIUS_KM
        return float(_LOG_EARTH_TARGET_AU * (ratio ** _LOG_SCALE_EXPONENT))
    if scaling == "marker":
        # Fixed small AU-sized marker — good enough until billboarded
        # markers arrive in a later iteration.
        return 0.01
    if scaling == "auto":
        # k_auto == 0.0 is the sentinel for "auto couldn't compute a
        # cohort-aware K" (no follow target). Fall back to log so the
        # user gets a sensible solar-system view rather than bodies at
        # physical scale (sub-pixel). For K >= 1.0 (which includes the
        # K=1 clamp from a tight flyby scenario), apply K * true radius
        # — at K=1 that's true scale, the right answer when bodies are
        # already as big as nearby separations allow.
        if k_auto == 0.0:
            ratio = radius_km / _EARTH_RADIUS_KM
            return float(_LOG_EARTH_TARGET_AU * (ratio ** _LOG_SCALE_EXPONENT))
        return k_auto * radius_km / _AU_KM
    raise ValueError(f"unknown scaling mode: {scaling!r}")


def _color_for(body_name: str) -> str:
    try:
        return resolve_body_constant(body_name).color_hex
    except UnknownBodyError:
        return _DEFAULT_COLOR


def _iau_rotation_matrices(
    *,
    alpha0_deg: float,
    delta0_deg: float,
    w0_deg: float,
    w_rate_deg_per_day: float,
    days_past_j2000: np.ndarray,
) -> np.ndarray:
    """Per-sample 3×3 rotation matrices taking body-fixed vectors to ICRF.

    IAU 2015 convention: ICRF→body = R_z(W) · R_x(90°−δ₀) · R_z(α₀+90°),
    so body→ICRF is the transpose of that product. Body-fixed frame:
    +Z = north pole, +X = prime meridian, +Y = 90° east. W(t) =
    `w0_deg` + `w_rate_deg_per_day` × Δd; α₀, δ₀ are J2000 values used
    as constants (sub-degree precession is invisible at viewer zoom).

    Returns shape (n_samples, 3, 3) — pre-computed once per body and
    stored on the viewer for cheap per-frame lookup.
    """
    n = days_past_j2000.shape[0]
    a = np.deg2rad(alpha0_deg + 90.0)
    d = np.deg2rad(90.0 - delta0_deg)
    w_arr = np.deg2rad(w0_deg + w_rate_deg_per_day * days_past_j2000)

    # R_z(α₀+90) and R_x(90-δ₀) are constant; build once.
    rz_alpha = np.array([
        [np.cos(a), -np.sin(a), 0.0],
        [np.sin(a),  np.cos(a), 0.0],
        [0.0,        0.0,       1.0],
    ])
    rx_delta = np.array([
        [1.0, 0.0,        0.0       ],
        [0.0, np.cos(d), -np.sin(d) ],
        [0.0, np.sin(d),  np.cos(d) ],
    ])
    rx_rz = rx_delta @ rz_alpha  # constant part

    # R_z(W) varies per sample. Build (n, 3, 3).
    cos_w = np.cos(w_arr)
    sin_w = np.sin(w_arr)
    rz_w = np.zeros((n, 3, 3), dtype=np.float64)
    rz_w[:, 0, 0] = cos_w
    rz_w[:, 0, 1] = -sin_w
    rz_w[:, 1, 0] = sin_w
    rz_w[:, 1, 1] = cos_w
    rz_w[:, 2, 2] = 1.0

    # ICRF→body per sample, then transpose to body→ICRF.
    icrf_to_body = np.einsum("nij,jk->nik", rz_w, rx_rz)
    return np.transpose(icrf_to_body, (0, 2, 1))


__all__ = ["Viewer", "Scaling"]
