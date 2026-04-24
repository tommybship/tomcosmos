import rebound


def make_simulation() -> rebound.Simulation:
    sim = rebound.Simulation()
    sim.units = ("km", "s", "kg")
    return sim
