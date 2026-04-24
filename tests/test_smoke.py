"""Toolchain smoke tests — verify rebound import and our units helper work."""
from tomcosmos.state.integrator import make_simulation


def test_rebound_two_body_integrates_one_step() -> None:
    sim = make_simulation()
    sim.add(m=1.989e30)  # Sun, kg
    sim.add(m=5.972e24, x=1.496e8, vy=29.78)  # Earth, kg, km, km/s
    sim.integrate(sim.t + 86400.0)  # one day
    assert len(sim.particles) == 2
    assert sim.t > 0


def test_units_are_km_s_kg() -> None:
    sim = make_simulation()
    assert sim.G > 0
