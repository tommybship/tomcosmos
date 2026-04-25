"""Rotating-frame transformation for tadpole / horseshoe analysis.

In the inertial (ICRF barycentric) frame, a Lagrange-point particle
co-rotates with the secondary, so its trajectory is a circle. To see
the *libration* — the small angular oscillation around L4/L5 that
defines a tadpole orbit — we need to rotate the scene so the secondary
sits at a fixed direction. In that rotating frame, L4 and L5 are
equilibria and the tadpole's angular wobble around them is visible.

Convention: primary at origin, secondary at (+R, 0, 0), orbital
angular momentum along +z. We rotate every body's position into this
frame at every sample. The transformation is:

    e_x(t) = (r_secondary(t) - r_primary(t)) / |..|
    e_z(t) = L_secondary(t) / |L|       (orbital angular momentum)
    e_y(t) = e_z × e_x

then for any other body:

    r_rot(t) = R(t) @ (r_inertial(t) - r_primary(t))

This module is import-light (numpy + pandas only) so the analysis
helpers don't pull in pyvista. The viewer's rotating-frame camera
mode reuses this same function to pre-compute body positions before
the scene is built.
"""
from __future__ import annotations

import numpy as np

from tomcosmos.io.history import StateHistory


def rotate_history_to_corotating(
    history: StateHistory, primary: str, secondary: str,
) -> dict[str, np.ndarray]:
    """Return per-body (n_samples, 3) position arrays in the frame
    co-rotating with `secondary` around `primary`.

    The output frame has primary at origin, secondary fixed at
    (+|r_sec − r_pri|, 0, 0), and the orbital plane in z=0. Every other
    body's trajectory is expressed in the same frame.

    Velocities are not transformed here — for tadpole-type analysis
    you usually want positions, and rotating velocities correctly
    requires the time derivative of the rotation matrix (Coriolis
    bookkeeping). Add a separate helper if velocity in the rotating
    frame is needed.
    """
    if primary == secondary:
        raise ValueError(f"primary and secondary must differ; got {primary!r}")
    bodies = set(history.df["body"].unique())
    for required in (primary, secondary):
        if required not in bodies:
            raise KeyError(f"{required!r} not in StateHistory")

    pivot = (
        history.df.sort_values(["sample_idx", "body"])
        .pivot(index="sample_idx", columns="body", values=["x", "y", "z"])
    )

    def _xyz(name: str) -> np.ndarray:
        return np.stack(
            [pivot[("x", name)], pivot[("y", name)], pivot[("z", name)]],
            axis=-1,
        )

    r_primary = _xyz(primary)
    r_secondary = _xyz(secondary)
    sep = r_secondary - r_primary
    R_sep = np.linalg.norm(sep, axis=1, keepdims=True)
    e_x = sep / R_sep

    # Orbital plane normal from finite-differenced velocity of the
    # secondary in the primary's frame. Two-sided diff for interior
    # samples, one-sided at endpoints; consistent with the cadence the
    # parquet was written at. Doesn't need to be ultra-precise — only
    # the *direction* is used to build the basis.
    rel = r_secondary - r_primary
    drel_dt = np.gradient(rel, axis=0)
    L = np.cross(rel, drel_dt)
    L_norm = np.linalg.norm(L, axis=1, keepdims=True)
    e_z = L / L_norm

    e_y = np.cross(e_z, e_x)

    out: dict[str, np.ndarray] = {}
    for name in bodies:
        r = _xyz(name) - r_primary
        # Project onto the (e_x, e_y, e_z) basis at each sample.
        out[name] = np.stack(
            [
                np.einsum("ij,ij->i", r, e_x),
                np.einsum("ij,ij->i", r, e_y),
                np.einsum("ij,ij->i", r, e_z),
            ],
            axis=-1,
        )
    return out


def angular_position_relative_to(
    rotated: dict[str, np.ndarray], particle: str, reference_angle_deg: float,
) -> np.ndarray:
    """Angular position of `particle` in the rotating frame's xy-plane,
    measured CCW from +x in degrees, then re-centered by subtracting
    `reference_angle_deg` so 0° means "at the reference angle".

    For Sun-Earth L4 with Earth at (+R, 0, 0): reference_angle_deg=60
    gives angles centered on L4's equilibrium, with positive values
    leading and negative trailing.

    Returns shape (n_samples,) wrapped to (-180, 180].
    """
    pts = rotated[particle]
    theta = np.rad2deg(np.arctan2(pts[:, 1], pts[:, 0]))
    delta = theta - reference_angle_deg
    # Wrap to (-180, 180].
    return ((delta + 180.0) % 360.0) - 180.0
