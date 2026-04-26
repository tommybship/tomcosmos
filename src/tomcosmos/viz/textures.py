"""Body textures — UV-mapped sphere meshes + image textures for the viewer.

PyVista bundles textures and UV-mapped meshes for the planets, the Sun,
and the Moon under `pv.examples.planets`. Earth's texture lives in the
top-level `pv.examples.load_globe_texture` (bundled, fully offline).
The other planets' textures are fetched on demand from pyvista's example
server — a one-time network call per body, cached locally afterward.

`load_for_body(name, radius_au)` returns `(mesh, texture)` for any body
tomcosmos knows how to texture, or `None` otherwise. The viewer uses
the textured pair when available and falls back to a plain solid-color
sphere when not (test particles, hypothetical bodies, anything not in
the table below).

Texture appearance is dependent on the viewer's scaling mode and zoom
level: at the default `log` scaling Earth is rendered at ~0.005 AU
display radius, so the texture is invisible from a default solar-system
view but becomes meaningful when the user zooms in or uses
`--follow earth` to track Earth from up close.

Adding more bodies: drop a new entry into `_TEXTURED_BODIES` mapping
the canonical body name to a (mesh-loader, texture-loader) pair.
Loaders are lazy callables so we don't pay the network/disk cost for
bodies the user never renders.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any


def _earth_texture() -> Any:
    import pyvista as pv
    return pv.examples.load_globe_texture()


def _earth_mesh(radius: float) -> Any:
    import pyvista as pv
    return pv.examples.planets.load_earth(radius=radius)


def _planet_textured_loader(
    mesh_loader: Callable[[float], Any],
    texture_downloader: Callable[..., Any],
) -> Callable[[float], tuple[Any, Any]]:
    """Wrap a (load_<planet>, download_<planet>_surface) pair into a
    single `(radius_au) -> (mesh, texture)` callable. The texture
    downloader is invoked with `texture=True` to get a `vtkTexture`
    rather than a numpy array."""
    def loader(radius: float) -> tuple[Any, Any]:
        return mesh_loader(radius=radius), texture_downloader(texture=True)
    return loader


# Map canonical body name → loader returning (mesh, texture).
# Earth is fully offline (the texture ships with pyvista). The other
# planets, the Sun, and the Moon use pyvista's downloadable textures —
# one-time network call per body, cached under pyvista's example dir.
def _build_textured_bodies() -> dict[str, Callable[[float], tuple[Any, Any]]]:
    import pyvista as pv

    p = pv.examples.planets

    return {
        "earth": lambda r: (_earth_mesh(r), _earth_texture()),
        "sun":     _planet_textured_loader(p.load_sun,     p.download_sun_surface),
        "mercury": _planet_textured_loader(p.load_mercury, p.download_mercury_surface),
        "venus":   _planet_textured_loader(p.load_venus,   p.download_venus_surface),
        "moon":    _planet_textured_loader(p.load_moon,    p.download_moon_surface),
        "mars":    _planet_textured_loader(p.load_mars,    p.download_mars_surface),
        "jupiter": _planet_textured_loader(p.load_jupiter, p.download_jupiter_surface),
        "saturn":  _planet_textured_loader(p.load_saturn,  p.download_saturn_surface),
        "uranus":  _planet_textured_loader(p.load_uranus,  p.download_uranus_surface),
        "neptune": _planet_textured_loader(p.load_neptune, p.download_neptune_surface),
        "pluto":   _planet_textured_loader(p.load_pluto,   p.download_pluto_surface),
    }


# Lazily-built so the import-time cost is just dict-of-lambdas, not
# `import pyvista` when this module is imported but never used.
_TEXTURED_BODIES: dict[str, Callable[[float], tuple[Any, Any]]] | None = None


def _registry() -> dict[str, Callable[[float], tuple[Any, Any]]]:
    global _TEXTURED_BODIES
    if _TEXTURED_BODIES is None:
        _TEXTURED_BODIES = _build_textured_bodies()
    return _TEXTURED_BODIES


def load_for_body(
    name: str, radius_au: float,
) -> tuple[Any, Any] | None:
    """Return `(mesh, texture)` for `name` if tomcosmos has a textured
    pair available, or `None` to signal "fall back to a plain sphere."

    `radius_au` is the desired display radius — typically the viewer's
    chosen scaling for the body, not its true physical radius. Returns
    a fresh mesh each call so callers can mutate `actor.position`
    without aliasing.

    For non-Earth bodies, the texture is downloaded from pyvista's
    example server on first call and cached locally afterward (network
    failure during the download surfaces as the underlying
    `requests`/`urllib` exception — caller can catch and fall back).
    """
    loader = _registry().get(name.lower())
    if loader is None:
        return None
    return loader(radius_au)
