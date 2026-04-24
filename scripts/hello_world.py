"""Sanity check: print Earth's heliocentric position at current time using skyfield."""
from skyfield.api import load


def main() -> None:
    ts = load.timescale()
    t = ts.now()
    planets = load("de440s.bsp")
    earth = planets["earth"]
    sun = planets["sun"]

    position = (earth - sun).at(t).position.km
    x, y, z = position
    print(f"Earth heliocentric position at {t.utc_iso()}:")
    print(f"  x = {x:+.3e} km")
    print(f"  y = {y:+.3e} km")
    print(f"  z = {z:+.3e} km")
    print(f"  |r| = {(x**2 + y**2 + z**2) ** 0.5:.3e} km")


if __name__ == "__main__":
    main()
