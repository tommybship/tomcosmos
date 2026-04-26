# tomcosmos

A Python solar system state simulator. JPL Horizons-grade asteroid propagation when configured for it; vanilla N-body for counterfactual scenarios. Reproducible Parquet outputs, declarative YAML scenarios, an interactive 3D viewer that scales to thousands of test particles.

## What it does

Two integration modes share one codebase, picked per scenario by `integrator.ephemeris_perturbers`:

- **Mode A** — wraps REBOUND with [ASSIST](https://github.com/matthewholman/assist), JPL's high-precision integrator. Gravity for the Sun, planets, Moon, Pluto, and 16 large asteroid perturbers comes from DE440 + sb441-n16 directly; GR (1PN) and J2 are baked in. Test particles only. Use for asteroid / NEO / mission propagation against the real solar system. Apophis at 30 days vs JPL Horizons: **72 m position, 30 µm/s velocity** ([test](tests/test_sbdb.py)).
- **Mode B** — vanilla REBOUND. Every massive body declared in the scenario; ICs come from skyfield reading NAIF SPK kernels. Optional GR via REBOUNDx's `gr` force. Use for Lagrange demos, Earth-Moon, Jupiter-Galileans, Earth-Mars Hohmann, hypothetical Planet 9, and any "what if" scenario.

Both modes share the scenario schema, IC seeding layer, viewer, and reproducible-output story. See [PLAN.md > Architecture: Mode A vs Mode B](PLAN.md#architecture-mode-a-vs-mode-b).

Other things tomcosmos does:
- Parses declarative YAML scenarios through a Pydantic schema; fails loudly on bad input before any integration starts.
- Writes trajectories to Parquet with embedded metadata (git SHA, kernel hashes, package versions, scenario YAML) so runs are reproducible.
- Detects Hill-sphere encounters post-hoc — even in Mode A where the major bodies aren't explicit `Body` entries (the analysis layer re-queries the ephemeris).
- Renders to 3D — orbit trails + colored planet spheres for small scenarios, single-PolyData point cloud for asteroid populations (1,000-body viewer hits ~4,500 frames/sec in our benchmarks).
- Bulk-ingests asteroids from JPL: `tomcosmos.targeting.sbdb` for catalog-scale studies, `tomcosmos.targeting.horizons` for live-truth scenarios. Disk cache makes Horizons re-runs free.

## Quickstart

```bash
# One-time: environment
conda env create -f environment.yml
conda activate tomcosmos

# Mode B baseline (no kernel-heavy deps): Sun + 8 planets, 10 years, WHFast.
tomcosmos fetch-kernels                              # ~32 MB DE440s
tomcosmos run    scenarios/sun-planets.yaml
tomcosmos view   runs/sun-planets-baseline__*.parquet

# Mode A asteroid: 30-day Apophis propagation against DE440.
tomcosmos fetch-kernels --include assist             # ~770 MB DE440 + sb441-n16
tomcosmos run    scenarios/apophis-30day.yaml
tomcosmos view   runs/apophis-30day__*.parquet
```

Bulk NEO demo (100 asteroids; the viewer's bulk-cohort renderer kicks in automatically above 20 test particles):

```bash
tomcosmos run    scenarios/neos-100.yaml
tomcosmos view   runs/neos-100__*.parquet
```

## CLI

| Command | What it does |
|---|---|
| `tomcosmos run <scenario>` | Integrate; write Parquet output. |
| `tomcosmos view <parquet>` | Open the 3D viewer with time slider. |
| `tomcosmos validate <scenario>` | Preflight: schema + ephemeris coverage + IC resolution. |
| `tomcosmos info <parquet>` | Print embedded run metadata (git SHA, kernel hashes, versions). |
| `tomcosmos fetch-kernels` | Download NAIF SPK kernels into `data/kernels/`. `--include {jupiter,saturn,neptune,pluto,mars,assist}` adds satellite-system kernels and the ASSIST set. |

## Scenarios

[`scenarios/`](scenarios/) holds canonical examples:
- `sun-planets.yaml` — Mode B baseline.
- `earth-moon.yaml`, `jupiter-galileans.yaml` — Mode B with satellite kernels.
- `sun-earth-l4-tadpole.yaml` — Mode B Lagrange demo.
- `earth-mars-hohmann.yaml` — Mode B Hohmann transfer with the runtime Lambert solver.
- `apophis-30day.yaml` — Mode A single-asteroid propagation.
- `neos-100.yaml` — Mode A bulk NEO demo (100 asteroids snapshotted from JPL Horizons).

Schema is documented by `tomcosmos.Scenario` ([src/tomcosmos/state/scenario.py](src/tomcosmos/state/scenario.py)); `tomcosmos validate` is the fastest way to learn what's accepted.

Key fields:
- `epoch` — ISO 8601 with explicit time scale, e.g. `"2026-04-23T00:00:00 TDB"`. `J2000` is an alias.
- `duration` — `"10 yr"`, `"50 day"`, `"3600 s"`, etc.
- `integrator.name` — `whfast` (fast, symplectic, fixed step), `ias15` (adaptive), or `mercurius` (hybrid for close encounters).
- `integrator.ephemeris_perturbers` — `false` (Mode B, default) or `true` (Mode A).
- `bodies[].ic.source` — `ephemeris` or `explicit`. (Mode A scenarios have empty `bodies`.)

## Python API

```python
from tomcosmos import Scenario, run, load_run

scenario = Scenario.from_yaml("scenarios/apophis-30day.yaml")
history  = run(scenario)                  # StateHistory: long-format DataFrame + metadata
history.to_parquet("runs/apophis.parquet")

# Round-trip
later   = load_run("runs/apophis.parquet")
apophis = later.body_trajectory("apophis")  # t_tdb, x, y, z, vx, vy, vz
events  = later.events                       # Hill-sphere encounters + Δv burns
```

The surface re-exported from `tomcosmos` is pinned. Off-surface until 1.0.

### Asteroid ingestion

Two paths, deliberately parallel — the user picks based on the question they're answering, **no default**:

```python
from astropy.time import Time
from tomcosmos.targeting import sbdb, horizons

# Catalog-style: "what JPL knows about this object, propagated forward in two-body gravity."
# Cheap; right when IC drift is bounded and uniform across the population.
orbit = sbdb.query("99942")
r_km, v_kms = sbdb.state_at_epoch(orbit, orbit.elements_epoch, ephemeris_source)

# Live-truth: "the asteroid's actual state at scenario epoch."
# JPL Horizons does the full N-body propagation up to the queried instant.
# Cached to data/cache/horizons_vectors.json so re-runs are free.
state = horizons.state_at_epoch("99942", Time("2026-04-26T00:00:00", scale="tdb"))
```

`scripts/build_neos_scenario.py --count N` regenerates `scenarios/neos-N.yaml` from the live JPL catalog.

## Accuracy

**Mode A** is the configuration to use when accuracy matters. Same physics as JPL Horizons (DE440 + sb441-n16 + GR + J2), so over typical mission timescales the residual is integrator step + roundoff:

| Asteroid | Span | Position vs Horizons | Velocity vs Horizons |
|---|---|---|---|
| Apophis | 30 days | < 100 m | < 1 µm/s |

(Mode A's `energy_rel_err` column is intentionally NaN — `sim.energy()` doesn't capture ASSIST's external forces, so the Hamiltonian-conservation diagnostic doesn't apply.)

**Mode B** is intentionally lower-fidelity — it's for scenarios where you've declared every body yourself and JPL agreement isn't the goal. Earth tracks the published ephemeris within ~2,000,000 km over one year for the default `sun-planets.yaml` (no moons, no GR by default). Adding `effects: [gr]` shrinks the Mercury residual by ~3,400 km over 10 yr, and adding moons further tightens the Earth envelope. Full envelope table in [PLAN.md > Accuracy envelope](PLAN.md#accuracy-envelope).

## Optional dependencies

The base install handles Mode B with no GR. Two optional extras unlock more:

- **`tomcosmos[reboundx]`** — REBOUNDx for Mode B optional forces (currently `gr` via `gr_full`; future Yarkovsky / radiation pressure for asteroid work). Linux/macOS gets it from PyPI; on Windows, install from [our patched fork](https://github.com/tommybship/reboundx/tree/windows-msvc-build) until the MSVC fixes merge upstream ([dtamayo/reboundx#137](https://github.com/dtamayo/reboundx/issues/137)).
- **`tomcosmos[assist]`** — ASSIST for Mode A. Linux/macOS gets it from PyPI; Windows from [our fork](https://github.com/tommybship/assist/tree/windows-msvc-build) (no upstream Windows port yet).

The conda `environment.yml` installs both optional extras from the forks by default — what you'd want if you're going to use either mode.

## Development

```bash
pytest                       # full suite (~22 s including assist + accuracy markers)
pytest -m "not viewer"       # skip viewer tests if no display
pytest -m ephemeris          # only ephemeris-touching tests
ruff check . && ruff format .
mypy src/tomcosmos
```

CI runs the non-ephemeris tier on push. See [.github/workflows/ci.yml](.github/workflows/ci.yml).

## License

MIT — see [LICENSE](LICENSE).
