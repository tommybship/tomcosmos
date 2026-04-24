# tomcosmos

A Python solar system state simulator: N-body integration with ephemeris-sourced initial conditions, reproducible Parquet outputs, and an interactive 3D viewer.

## What it does

- Integrates the solar system (Sun + planets, moons and test particles in later milestones) from real ephemeris ICs using [REBOUND](https://rebound.readthedocs.io/).
- Parses declarative YAML scenarios through a Pydantic schema; fails loudly on bad input before any integration starts.
- Writes trajectories to Parquet with embedded metadata (git SHA, kernel hashes, package versions, scenario YAML) so runs are reproducible.
- Renders the result as a 3D scene: orbit trails, colored planets, time slider.

## Status

**M1 — Sun + 8 planets, validated.** WHFast energy conservation holds at < 1e-10 relative over 10 years. Earth tracks ephemeris within ~2,000,000 km over one year — the bulk of that drift is the physics we deliberately skip (moons, GR, asteroid belt), not integration error. See [PLAN.md](PLAN.md) for the full roadmap through M6 and the measured accuracy envelope.

## Quickstart

```bash
# One-time: environment
conda env create -f environment.yml
conda activate tomcosmos

# One-time: fetch the DE440s ephemeris (~32 MB)
tomcosmos fetch-kernels

# Integrate the baseline scenario (Sun + 8 planets, 10 years, WHFast at 1-day step)
tomcosmos run scenarios/sun-planets.yaml

# Open the 3D viewer on the run you just produced
tomcosmos view runs/sun-planets-baseline__*.parquet
```

Other CLI commands:
- `tomcosmos validate <scenario>` — preflight a scenario (schema + ephemeris coverage + body lookups).
- `tomcosmos info <run.parquet>` — print the embedded run metadata.
- `tomcosmos version` — package version.

## Scenarios

YAML under `scenarios/` describes what to simulate. [`scenarios/sun-planets.yaml`](scenarios/sun-planets.yaml) is the canonical baseline. The schema is documented by `tomcosmos.Scenario` (see `src/tomcosmos/state/scenario.py`); `tomcosmos validate` is the fastest way to learn what's accepted.

Key fields:
- `epoch` — ISO 8601 with explicit time scale, e.g. `"2026-04-23T00:00:00 TDB"`. `J2000` is an alias.
- `duration` — `"10 yr"`, `"50 day"`, `"3600 s"`, etc.
- `integrator` — `whfast` (fast, symplectic, fixed step), `ias15` (adaptive, high precision), or `mercurius` (hybrid for close encounters).
- `bodies[].ic.source` — `ephemeris` (queried from the kernel) or `explicit` (you provide `r` and `v`, optionally with a non-ICRF frame).

## Python API

```python
from tomcosmos import Scenario, run, load_run

scenario = Scenario.from_yaml("scenarios/sun-planets.yaml")
history = run(scenario)                      # integrates, returns StateHistory
history.to_parquet("runs/mine.parquet")      # persist

later = load_run("runs/mine.parquet")        # round-trips metadata and scenario
earth = later.body_trajectory("earth")       # pandas DataFrame, t_tdb + x,y,z,vx,vy,vz
```

The surface exported from `tomcosmos` is pinned; anything not re-exported there is off-surface until 1.0.

## Accuracy notes

This is a **learning-grade** simulator, not a mission-planning tool. It doesn't model GR, non-gravitational forces, tides, or planetary rotation (see [PLAN.md > Non-goals](PLAN.md#non-goals)). The practical ceiling is ~1e-4 relative because we use `G × mass` rather than JPL's `GM`.

Measured envelope (same-machine reproducibility, baseline sun-planets scenario):

| Span | Earth vs ephemeris | Mercury vs ephemeris |
|---|---|---|
| 1 year | < 2e6 km | < 7e5 km |
| 10 years | < 2e7 km | < 5e6 km |
| 100 years | < 2e8 km | < 5e7 km |

These are regression gates, not aspirational numbers — the drift comes mostly from missing physics (moons, GR), not the integrator. Adding the Moon to a scenario roughly halves the Earth envelope. The full table lives in [PLAN.md > Accuracy envelope](PLAN.md#accuracy-envelope).

## Development

```bash
pytest                    # unit + physics-invariant tests (fast)
pytest -m ephemeris       # tolerance tests against the kernel (slower)
ruff check . && ruff format .
mypy src/tomcosmos
```

CI runs the non-ephemeris tier on push. See `.github/workflows/ci.yml`.

## License

MIT — see [LICENSE](LICENSE).
