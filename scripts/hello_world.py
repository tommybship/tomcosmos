"""Sanity check: print Earth's heliocentric position at current time using skyfield.

Loads the DE440s kernel from `tomcosmos.config.kernel_dir()`
(default `./data/kernels`; override with TOMCOSMOS_KERNEL_DIR). If the
kernel isn't there yet, skyfield will download it on first run.
"""
from skyfield.api import Loader

from tomcosmos.config import kernel_dir


def main() -> None:
    d = kernel_dir()
    d.mkdir(parents=True, exist_ok=True)
    loader = Loader(str(d))
    ts = loader.timescale()
    t = ts.now()
    planets = loader("de440s.bsp")
    earth = planets["earth"]
    sun = planets["sun"]

    position = (earth - sun).at(t).position.km
    x, y, z = position
    print(f"Earth heliocentric position at {t.utc_iso()}:")
    print(f"  x = {x:+.3e} km")
    print(f"  y = {y:+.3e} km")
    print(f"  z = {z:+.3e} km")
    print(f"  |r| = {(x**2 + y**2 + z**2) ** 0.5:.3e} km")
    print(f"(kernel: {d / 'de440s.bsp'})")


if __name__ == "__main__":
    main()
