# Solar System State Simulator — Plan

## Contents
1. [Context](#context)
2. [Non-goals](#non-goals)
3. [Project location and layout](#project-location-and-layout)
3. [Design conventions](#design-conventions)
4. [Dependency plan](#dependency-plan) — `environment.yml` + `pyproject.toml` sketches
5. [GitHub integration](#github-integration)
6. [Architecture (three decoupled layers)](#architecture-three-decoupled-layers)
7. [Iterative milestones](#iterative-milestones-each-runnable-end-to-end) — M0 → M6 with exit criteria
8. [Developer workflow + CLI](#developer-workflow--cli) — including `run()` orchestration flow
9. [Testing strategy](#testing-strategy)
10. [Special IC computation](#special-ic-computation) — Lagrange, Keplerian, explicit
11. [Reproducibility + diagnostics](#reproducibility--diagnostics)
12. [Critical files](#critical-files-rough-layout-when-building)
13. [Reuse](#reuse--do-not-build-these-yourself)
14. [Verification](#verification)
15. [Operational behaviors](#operational-behaviors) — close encounters, divergence, checkpointing, perf targets, error handling
16. [Open questions for later](#open-questions-for-later-not-blocking)
17. [Glossary](#glossary)

## Context
Build a Python-based solar system simulator with **learning-grade accuracy** — tight enough that the physics reads clearly (Lagrange points emerge naturally from N-body rather than sphere-of-influence approximations; Kepler's third law recoverable from measured periods; energy bounded under symplectic integration), but explicitly not mission-planning-grade. See Non-goals for what we're deliberately leaving out. 3D visualization, with scope that grows from Sun+planets up to spacecraft-like test particles and small bodies. Motivation is learning orbital mechanics while having something that feels like a real tool. Shareability matters; packaging later is acceptable.

## Non-goals

Pinning these explicitly so scope doesn't creep across milestones:
- **General relativity.** No GR corrections. Mercury's perihelion will precess at Newtonian rate, ~0.43 arcsec/century short of observed. `REBOUNDx.gr` is a future option, not a commitment.
- **Non-gravitational forces.** No solar radiation pressure, Yarkovsky, outgassing, atmospheric drag. Small-body trajectories (M5) are pure-gravity approximations.
- **Planetary rotation / orientation dynamics.** Spin axes are available as data for visualization, but no torque integration. No tidal dissipation.
- **Actual mission design.** No Lambert solvers, trajectory optimization, or targeting. M4 "spacecraft" scenarios are for visualizing maneuvers, not planning them.
- **Byte-identical cross-platform reproducibility.** Floating-point associativity under different BLAS/CPU combinations makes this intractable. Same-machine reruns agreeing to ~1e-10 is the bar.
- **Launch / atmospheric phase.** M4 probes start already in heliocentric orbit; we don't model ascent, Earth-departure maneuvers relative to Earth's rotating frame, or anything requiring geocentric sub-day timesteps.

## Project location and layout
- **On disk**: `C:\git\tommybship\tomcosmos` — all paths in this plan are relative to this directory.
- **Layout**: Python **"src layout"** — importable code lives under `src/tomcosmos/`, not at the repo root. Standard for modern Python packaging: prevents accidental imports of the in-tree source when running tests, and is what `pip install -e .` expects.
- **Environment**: one conda env named `tomcosmos`, built from `environment.yml`. If this project later gets published, runtime deps move to `pyproject.toml` `[project.dependencies]` and the env splits cleanly at that point — no refactor needed now.

Top-level tree:
```
C:\git\tommybship\tomcosmos\
├── environment.yml
├── pyproject.toml          # project metadata, [project.scripts] = tomcosmos CLI, editable install
├── README.md
├── .gitignore              # data/kernels/, runs/, .venv/, __pycache__/, etc.
├── src/
│   └── tomcosmos/
│       ├── __init__.py     # public API: Scenario, run, load_run, StateHistory
│       ├── cli.py          # typer commands
│       ├── config.py       # runtime paths (kernel dir, runs dir), defaults
│       ├── constants.py    # body palette, physical constants, SPICE ID map
│       ├── exceptions.py   # ScenarioValidationError, IntegratorDivergedError, etc.
│       ├── state/          # "what's being simulated" — physics model
│       │   ├── __init__.py
│       │   ├── scenario.py    # Pydantic models (Scenario, Body, TestParticle, ...)
│       │   ├── ephemeris.py   # EphemerisSource ABC + SkyfieldSource/SpiceSource
│       │   ├── ic.py          # special IC computation (Lagrange, Keplerian, explicit)
│       │   ├── integrator.py  # REBOUND wrapper, integrator selection from scenario
│       │   ├── events.py      # Δv events, encounter callbacks
│       │   └── frames.py      # frame conversions (ICRF ↔ ecliptic ↔ rotating)
│       ├── io/             # persistence + diagnostics
│       │   ├── __init__.py
│       │   ├── history.py     # StateHistory, Parquet read/write with embedded metadata
│       │   └── diagnostics.py # structlog setup, run-metadata capture
│       ├── analysis/       # post-hoc helpers on top of StateHistory
│       │   ├── __init__.py
│       │   ├── orbital_elements.py  # osculating elements from state vectors
│       │   ├── encounters.py        # close-approach detection
│       │   └── metrics.py           # energy/angular-momentum traces, period extraction
│       └── viz/            # rendering
│           ├── __init__.py
│           ├── scene.py       # shared scene construction (used by both backends)
│           ├── pyvista_viewer.py
│           └── web.py         # M6 trame app
├── scripts/                # fetch_kernels.py, hello_world.py
├── scenarios/              # *.yaml scenario configs
├── tests/                  # mirrors src/tomcosmos/ layout
├── data/
│   └── kernels/            # SPICE / ephemeris files (gitignored)
└── runs/                   # Parquet outputs + logs (gitignored)
```

**No `utils/` package.** Things without a home usually indicate missing structure; force them into a named module instead.

## Design conventions

Python is multi-paradigm; pick the right tool per job rather than forcing OO everywhere.

### Data objects — Pydantic models and dataclasses
Structured values use Pydantic v2 models (for scenarios — validation is worth it) or `@dataclass(frozen=True)` (for internal values where validation is overkill). Examples:
- `Scenario`, `Body`, `TestParticle`, `DeltaVEvent`, `IntegratorConfig`, `OutputConfig` — Pydantic (parsed from YAML, need validation).
- `StateHistory`, `RunMetadata` — dataclass; internal, produced by trusted code.

### Pure functions for physics
Stateless computations are just functions. No class needed:
- `compute_lagrange_position(primary_state, secondary_state, point, mass_ratio) → (r, v)`
- `solve_kepler(M, e) → E`
- `icrf_to_ecliptic(r) → r`
- `parse_duration("10 yr") → astropy.units.Quantity`

Classes exist to hide state. Pure functions don't have state to hide.

### Classes where state earns its keep
Wrap external stateful resources and things with lifecycle:
- `Integrator` — wraps a `rebound.Simulation`, owns the timestep, tracks energy-error history during integration. Not an ABC; internal dispatch by integrator name in the scenario.
- `Viewer` — wraps `pyvista.Plotter`, holds camera state, trail buffers, current sample index.

### ABCs for strategy interfaces
Only when there are genuinely interchangeable implementations. Currently one:
- `EphemerisSource` (ABC) with `SkyfieldSource` and `SpiceSource`. M1 uses skyfield; M2 switches to spiceypy for satellite kernels without changing any caller. The ABC defines `query(body_id, epoch) → (r, v)` and `available_bodies()`.

Don't pre-emptively ABC things with one implementation ("what if we swap integrators" — REBOUND already abstracts this; double-wrapping adds nothing).

### Avoid deep inheritance
`Body → Planet → RockyPlanet → Earth` is a classic physics-sim anti-pattern. Prefer composition: a `Body` has attributes (mass, radius, optional spin); type-specific behavior lives in functions that branch on those attributes, not in subclass methods.

### Module boundaries
- `state/` doesn't import from `viz/` or `io/` (physics is independent of how you render or persist it).
- `analysis/` reads `StateHistory`; doesn't touch `state/` internals.
- `viz/` reads `StateHistory`; doesn't touch `state/` internals.
- `io/` knows about `StateHistory` and `Scenario`; nothing above it.

This keeps the dependency graph acyclic and makes the physics unit-testable without pyvista or Parquet in the loop.

## Dependency plan

Versions pinned to major-version ranges. Conda-forge is the source for everything; conda-forge packages have consistent binary compatibility (no separate pip install layer for scientific deps).

### Runtime dependencies (needed for `tomcosmos run` and `view`)

| Package | Version pin | Purpose | Introduced in |
|---|---|---|---|
| `python` | `=3.12` | Interpreter | M0 |
| `numpy` | `>=1.26,<3` | Arrays, vector math | M0 |
| `scipy` | `>=1.11,<2` | `optimize.brentq` for Lagrange quintic, Newton's method for Kepler's eq | M0 |
| `rebound` | `>=4,<5` | N-body integrator (WHFast, IAS15, MERCURIUS) | M0 |
| `astropy` | `>=6,<8` | Time scales (TDB/UTC/TT), units, constants, coordinate frames | M0 |
| `skyfield` | `>=1.48,<2` | Ephemeris on-ramp for M1 | M1 |
| `spiceypy` | `>=6,<8` | SPICE kernel access for satellite ephemerides | M2 |
| `astroquery` | `>=0.4.6` | JPL Small-Body Database queries | M5 |
| `pydantic` | `>=2.5,<3` | Scenario schema validation | M0 |
| `pyyaml` | `>=6.0,<7` | YAML parsing for scenarios | M0 |
| `typer` | `>=0.12,<1` | CLI framework | M0 |
| `structlog` | `>=24,<26` | Structured JSON logging | M0 |
| `pyarrow` | `>=15,<20` | Parquet read/write | M0 |
| `pandas` | `>=2.2,<3` | `StateHistory` DataFrame surface, quick-look plots | M0 |
| `pyvista` | `>=0.43,<1` | 3D visualization (VTK under the hood) | M1 |
| `trame` | `>=3.5,<4` | Web viewer (pyvista scenes served to browser) | M6 |

### Dev-only dependencies

| Package | Version pin | Purpose |
|---|---|---|
| `pytest` | `>=8,<9` | Test runner |
| `pytest-xdist` | `>=3.5,<4` | Parallel test execution |
| `pytest-mpl` | `>=0.17,<1` | Viewer snapshot tests |
| `mypy` | `>=1.10` | Static type checking (strict mode for `src/tomcosmos/state/`) |
| `ruff` | `>=0.5` | Linting + formatting (replaces black, isort, flake8) |

### Not pulled in (deliberate non-choices)
- **`poliastro`** — great library, but its upstream is in flux (archived/forked); don't build on an unstable base. REBOUND + astropy + hand-rolled IC helpers cover what we need.
- **`h5py`** — not needed while Parquet is enough. Add only if M5 needs streaming writes for very long runs.
- **`polars`** — faster than pandas for some workloads, but pandas is adequate for `StateHistory` sizes we'll hit. Swap later if `analysis/` becomes a bottleneck.
- **`numba` / `cython`** — REBOUND is already C; your Python code won't be the bottleneck. Don't optimize prematurely.
- **`matplotlib`** — shipped with pandas/astropy for quick-look plots; no pin needed, it'll come in transitively.

### `environment.yml` sketch

```yaml
name: tomcosmos
channels: [conda-forge]
dependencies:
  - python=3.12
  # runtime
  - numpy>=1.26,<3
  - scipy>=1.11,<2
  - rebound>=4,<5
  - astropy>=6,<8
  - skyfield>=1.48,<2
  - spiceypy>=6,<8
  - astroquery>=0.4.6
  - pydantic>=2.5,<3
  - pyyaml>=6.0,<7
  - typer>=0.12,<1
  - structlog>=24,<26
  - pyarrow>=15,<20
  - pandas>=2.2,<3
  - pyvista>=0.43,<1
  - trame>=3.5,<4
  # dev
  - pytest>=8,<9
  - pytest-xdist>=3.5,<4
  - pytest-mpl>=0.17,<1
  - mypy>=1.10
  - ruff>=0.5
  - pip
  - pip:
      - -e .              # editable install of the tomcosmos package
```

### `pyproject.toml` sketch

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "tomcosmos"
version = "0.0.1"
description = "Solar system state simulator."
requires-python = ">=3.12,<3.13"
readme = "README.md"
license = { text = "MIT" }
authors = [{ name = "Tommy Blankinship" }]
# Runtime deps listed here so `pip install .` works even without conda.
# Versions match environment.yml pins.
dependencies = [
  "numpy>=1.26,<3",
  "scipy>=1.11,<2",
  "rebound>=4,<5",
  "astropy>=6,<8",
  "skyfield>=1.48,<2",
  "spiceypy>=6,<8",
  "astroquery>=0.4.6",
  "pydantic>=2.5,<3",
  "pyyaml>=6.0,<7",
  "typer>=0.12,<1",
  "structlog>=24,<26",
  "pyarrow>=15,<20",
  "pandas>=2.2,<3",
  "pyvista>=0.43,<1",
  "trame>=3.5,<4",
]

[project.optional-dependencies]
dev = [
  "pytest>=8,<9",
  "pytest-xdist>=3.5,<4",
  "pytest-mpl>=0.17,<1",
  "mypy>=1.10",
  "ruff>=0.5",
]

[project.scripts]
tomcosmos = "tomcosmos.cli:app"

[tool.hatch.build.targets.wheel]
packages = ["src/tomcosmos"]

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP", "SIM", "N"]
ignore = ["E501"]  # line length handled by formatter

[tool.mypy]
python_version = "3.12"
strict = false  # start loose
# strict for the physics core — expect friction with astropy.units.Quantity
# arithmetic; keep return-type inference lenient so Quantity ops don't poison
# every signature:
[[tool.mypy.overrides]]
module = "tomcosmos.state.*"
strict = true
warn_return_any = false

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
  "physics: physics-invariant tests (medium speed)",
  "ephemeris: ephemeris-agreement tests (slow, network/kernel access)",
  "viewer: viewer snapshot tests (require display or offscreen GL)",
]
addopts = "-ra --strict-markers"
```

## GitHub integration

### Repo setup
- Create `tommybship/tomcosmos` on github.com (empty repo — no README/LICENSE/gitignore template; we'll commit those locally first).
- Local: `git init` → first commit of scaffolding → `git remote add origin git@github.com:tommybship/tomcosmos.git` → `git push -u origin main`.
- Branch model: **main + short feature branches**. Commit directly to main for scaffolding and one-off fixes; open a PR-to-self for anything spanning multiple commits or touching the integrator / IC math (reading your own diff is a cheap forcing function). CI runs on push-to-main and PR alike, so either path gets the same gating.
- Commit style: short imperative subject lines (`add Lagrange IC for L4/L5`, `pin rebound to v4`). Conventional Commits is optional — adopt only if you'll use it consistently.

### Repo hygiene
- `.gitignore` covers: `data/kernels/`, `runs/`, `.venv/`, `__pycache__/`, `*.egg-info/`, `.pytest_cache/`, `.mypy_cache/`, `.DS_Store`, `.vscode/` (unless you want to share settings).
- **Large files**: GitHub rejects pushes > 100 MB per file. The bundled test ephemeris `de440s.bsp` is ~32 MB — fine for regular git. Full `de440.bsp` (~114 MB) stays out of the repo entirely (downloaded on demand via `fetch-kernels`).
- **Secrets**: none expected, but add a commit hook check for `AWS_`, `SECRET`, `TOKEN`, etc. via `detect-secrets` if paranoid. Probably YAGNI for a personal project.

### CI — GitHub Actions
Worth adding early (M0 or M1). One workflow, `.github/workflows/ci.yml`:
- Triggers: `push` to main, `pull_request` against main.
- Matrix: `ubuntu-latest` primary, Python 3.12. `windows-latest` optional — REBOUND + pyvista on Windows GitHub runners have had historical flakiness (esp. headless GL for viewer tests); add only if it stays green. Windows-specific regressions are covered by local dev-machine verification regardless.
- Steps:
  1. Checkout
  2. Setup conda (use `conda-incubator/setup-miniconda@v3` with `environment-file: environment.yml`)
  3. `pip install -e .`
  4. `ruff check .`
  5. `mypy src/tomcosmos/state`
  6. `pytest -n auto -m "not ephemeris and not viewer"` (skip slow/headless-unfriendly tests on CI)
- Optional secondary workflow: `weekly.yml` runs the ephemeris-agreement tests on a cron (`schedule: "0 6 * * 1"`), since those pull data and are slow.

### README structure (sketch)
Pin the outline early; flesh out per milestone:
```
# tomcosmos
One-line description.

## What it does
## Status (what M is it on)
## Quickstart (3-command install + run a scenario + view)
## Scenarios (link to scenarios/ examples)
## Accuracy notes (what we model vs. what JPL models — set expectations)
## Development (conda env, tests, linting)
## License
```

### LICENSE
Pick one now even if the repo is private; makes "public someday" a one-step decision. **MIT** is the low-friction standard for personal tool-projects. Drop `LICENSE` at the repo root in M0.

### Issues / Projects
Use GitHub Issues as your milestone tracker. Create one issue per "M2 — Major moons" etc.; close when the exit criteria are met. Don't bother with Projects boards for a solo project unless you already love kanban.

## Architecture (three decoupled layers)

### 1. State layer — ephemeris + N-body, cleanly separated
- **`spiceypy`** for ephemeris lookups and initial conditions (loads NAIF SPICE kernels — DE440 planetary, satellite kernels for moons). Pragmatic on-ramp: start with **`skyfield`** in M1 (pip-installable, no kernel wrangling), migrate to spiceypy in M2 when you need satellite data.
- **`REBOUND`** as the N-body integrator. Integrator per milestone detailed below.
- **`astropy.time`** for time scales (TDB for dynamics, UTC only for display — don't mix them). **`astropy.units` / `astropy.constants`** to keep dimensions honest.
- One frame throughout: **ICRF, solar-system barycenter origin.** SPICE gives ICs in this frame; REBOUND runs in it; transform only at the render step.

#### Integrator choice and timesteps (per milestone)
REBOUND provides three integrators we'll use. The rule of thumb: timestep ≤ ~5% of the shortest orbital period in the system.

| Milestone | Integrator | Timestep | Energy-error target | Notes |
|---|---|---|---|---|
| M1 (planets) | **WHFast** | 1 day | `|ΔE/E|` bounded ≤ 1e-10 | Mercury's period is 88 days → 1 day = ~1.1% of period, safely stable. Symplectic → energy is bounded forever; drift is numerical roundoff only. |
| M2 (moons) | **IAS15** | adaptive | ≤ 1e-14 per step (machine precision) | Io's period is 1.77 days — WHFast at 1 day blows up. Options were (a) WHFast at 0.05 day (20× cost), (b) IAS15 adaptive. IAS15 wins: handles Io and Neptune in one run without fixed-step tuning. |
| M3 (Lagrange, test particles) | **IAS15** | adaptive | ≤ 1e-14 per step | Decade-long L4 tadpole orbits need near-machine-precision energy conservation to not wash out the libration signal. IAS15's adaptive step handles this. |
| M4 (spacecraft, close encounters) | **MERCURIUS** | ~1 day base | ≤ 1e-8 | WH outside Hill radii, IAS15 inside — gives WHFast speed + close-encounter correctness. The Hill-radius switchover parameter (`ri_mercurius.r_crit_hill`, default 3) is what to tune. **The 1-day base step assumes heliocentric probes** — a geocentric orbit has period ≪ 1 day and is out of scope (see Non-goals). M4 scenarios start with the probe already in solar orbit. |
| M5 (small bodies) | **WHFast** with test-particle type 0 | 1 day | ≤ 1e-10 on massive bodies | Tens of thousands of massless particles: set `N_active` to the count of massive bodies and `testparticle_type = 0` so particles don't perturb planets. Force calculation scales ~linearly in N. |

**Operational guidance**
- **Log energy error every output cadence.** Persist it in the state-history file. Drift = wrong timestep or wrong integrator; oscillation = normal.
- **Validate M1 against ephemeris.** After 1 year of WHFast, Earth should match skyfield/SPICE to <1000 km. After 100 years, drift of 10s of thousands of km is expected (you're not modeling GR, asteroid-belt perturbations, or tides) — that's physics, not a bug.
- **Never switch integrators mid-run** silently. If a scenario forces a switch (e.g., spacecraft added), the scenario file should name MERCURIUS explicitly.
- **WHFast requires heliocentric (or Jacobi) coordinates internally** but REBOUND handles this automatically; you feed barycentric state vectors, REBOUND converts on `sim.integrate()`.

### 2. Scenario layer — what's being simulated
- **Time**: all epochs stored and displayed as ISO 8601 with an explicit time scale (`2026-04-23T00:00:00 TDB`). Internal sim time is TDB. astropy.time handles conversions. J2000 (`2000-01-01T12:00:00 TDB`) is available as a named alias for convenience.
- **Format**: **YAML** (PyYAML). Comments matter — scenarios are the thing you'll read most often. Schema validated by a Pydantic model (`src/tomcosmos/state/scenario.py`) so invalid configs fail at load, not mid-simulation.
- **Data model**:
  - `Body { name, mass_kg?, radius_km?, ic: { source: ephemeris|explicit, spice_id?, r?, v? }, spin? }` — `mass_kg` and `radius_km` are optional; if omitted, resolved from `constants.py` by `name` or `spice_id` (see Body constants below).
  - `TestParticle { name, ic: { type: explicit|lagrange|keplerian, ... } }`
  - `IntegratorConfig { name: whfast|ias15|mercurius, timestep?, divergence_threshold?, r_crit_hill? }` — `timestep` required for fixed-step integrators, omitted for adaptive. `divergence_threshold` defaults per-integrator (see Operational behaviors).
  - `OutputConfig { format: parquet, cadence, path?, checkpoint?: bool }` — `path` optional; if omitted, defaults to `runs/<scenario_name>__<run_started_utc>.parquet` (see Output paths below).
  - `Scenario { name, epoch, duration, integrator, bodies, test_particles?, output }`

#### Output paths and overwrite behavior
- **Default path is timestamped** (`runs/<scenario_name>__<utc_iso_basic>.parquet`, e.g. `runs/sun-planets-baseline__20260424T153000Z.parquet`). Two runs of the same scenario don't silently clobber each other.
- **Explicit `output.path` or `--output <path>` overrides the default** but refuses to overwrite an existing file unless `--overwrite` is passed. This protects against the common mistake of re-running a scenario whose output you were about to analyze.
- The event-log path mirrors the state-history path (`<basename>.events.parquet`); the log file mirrors it too (`<basename>.log`).

#### Body constants (where mass and radius come from)
- `src/tomcosmos/constants.py` ships a canonical table for the Sun, 8 planets, Pluto, and major moons — sourced from NASA fact sheets and cross-checked against astropy's `solar_system_ephemeris`. Values are authoritative-enough for our purposes (four significant figures; real GM values are known to more).
- Resolution order for a `Body`:
  1. Explicit `mass_kg` / `radius_km` in the scenario YAML wins.
  2. Lookup by `spice_id`, then by lowercased `name`.
  3. If neither present → `ScenarioValidationError` at load with a list of suggestions.
- Test particles (`TestParticle`) are always massless (`m = 0` in REBOUND); a radius of 1 km is used for collision detection only.
- Why not pull from SPICE PCK files? We could, but it adds a kernel dependency to M1 for data that barely changes. Revisit in M2 when SPICE is already loaded — at that point, optionally verify `constants.py` against PCK values in a test.

#### Ephemeris time-range validation
- DE440 covers 1550-06-16 to 2650-01-25. Asking for epochs outside this range is a user error that should fail at scenario load, not mid-integration.
- `Scenario` Pydantic validator checks that `epoch + duration` falls within the loaded ephemeris source's `time_range()`. `EphemerisSource.time_range()` is part of the ABC contract — each backend reports its own coverage.
- For extrapolation beyond the ephemeris range (e.g., long stability studies), the scenario can set `bodies[*].ic.source: explicit` and carry forward a state vector snapshot from a shorter bounded run.
- **Example** (`scenarios/sun-planets.yaml`):

  ```yaml
  name: sun-planets-baseline
  epoch: "2026-04-23T00:00:00 TDB"
  duration: "10 yr"
  output:
    format: parquet
    cadence: "1 day"
    path: "runs/sun-planets-baseline.parquet"
  integrator:
    name: whfast
    timestep: "1 day"
  bodies:
    - { name: Sun,     spice_id: 10,  ic: { source: ephemeris } }
    - { name: Mercury, spice_id: 199, ic: { source: ephemeris } }
    - { name: Venus,   spice_id: 299, ic: { source: ephemeris } }
    - { name: Earth,   spice_id: 399, ic: { source: ephemeris } }
    - { name: Mars,    spice_id: 499, ic: { source: ephemeris } }
    - { name: Jupiter, spice_id: 599, ic: { source: ephemeris } }
    - { name: Saturn,  spice_id: 699, ic: { source: ephemeris } }
    - { name: Uranus,  spice_id: 799, ic: { source: ephemeris } }
    - { name: Neptune, spice_id: 899, ic: { source: ephemeris } }
  test_particles: []   # populated in M3+
  ```

- **Example** (Lagrange demo, M3):

  ```yaml
  name: sun-earth-l4-tadpole
  epoch: "2026-04-23T00:00:00 TDB"
  duration: "50 yr"
  integrator: { name: ias15 }   # adaptive, no timestep needed
  bodies:
    - { name: Sun,   spice_id: 10,  ic: { source: ephemeris } }
    - { name: Earth, spice_id: 399, ic: { source: ephemeris } }
  test_particles:
    - name: L4-probe
      ic: { type: lagrange, primary: Sun, secondary: Earth, point: L4 }
  ```

- **Output**: state history (times × bodies × 6) to **Parquet** (columnar, pandas/polars-friendly, good for slicing by body or time). Switch to HDF5 only if you later stream partial writes during very long runs. Separating sim from render means you re-render without re-simulating.

#### `StateHistory` schema (pinned now — viz, analysis, and io all depend on it)
Long format, one row per (time × body). Pandas/pyarrow friendly, slices cleanly by body or time range.

| Column | Type | Units | Notes |
|---|---|---|---|
| `sample_idx` | `int64` | — | 0-indexed sample number; monotonically increasing. Primary key with `body` |
| `t_tdb` | `float64` | seconds from epoch | Simulation time. Epoch itself lives in file metadata, not rows |
| `body` | `string` (dict-encoded) | — | Body name; dict-encoded in Parquet for compactness |
| `x`, `y`, `z` | `float64` | km | Position in ICRF, SSB-origin |
| `vx`, `vy`, `vz` | `float64` | km/s | Velocity in ICRF |
| `terminated` | `bool` | — | True for samples after an impact/escape event; position columns are NaN |
| `energy_rel_err` | `float64` | — | `|ΔE/E|` at this sample; same value repeated across bodies for a given `sample_idx` (one-pass write; cheap) |

Rationale for long vs wide:
- Long format (one row per body per time) makes "plot Earth's trajectory" a single filter; wide format (one row per time, columns like `earth_x`, `mars_x`, ...) is awkward once the body set is dynamic (test particles, M5 small bodies).
- Parquet's dictionary encoding on `body` makes the storage overhead of long format essentially zero.
- Groupby-by-body queries are fast with pyarrow's predicate pushdown.

File metadata (Parquet key/value, not per-row):
- All fields from the "Reproducibility + diagnostics" section (scenario YAML, git SHA, kernel hashes, versions, wall-clock, integrator config).
- `epoch_iso` — the scenario epoch as ISO 8601 TDB string.
- `body_constants` — JSON blob of the `mass_kg`, `radius_km`, `color` used for each body, so the viewer doesn't need to re-resolve from `constants.py`.

Energy-error diagnostics live in the `energy_rel_err` column only — not also as a file-metadata array. Parquet's run-length encoding makes the repeated-across-bodies values essentially free, and keeping it columnar means `pandas.read_parquet(...).groupby("sample_idx").energy_rel_err.first()` is the canonical query. One source of truth.

Event log (separate Parquet file, `runs/<name>.events.parquet`):
- Rows: `(t_tdb, kind, particle, body?, detail)`. `kind` ∈ `{delta_v, hill_enter, hill_exit, impact, divergence}`. Keeps the main state history dense and typed.

### 3. Visualization layer — pyvista now, trame for web later
- **`pyvista`** desktop 3D viewer reads state history, renders bodies with orbit trails, supports time scrubbing and camera targeting. **`pyvista` + `trame`** exposes the same scene as a web app — "share it" and "package it" without porting to three.js.

#### Body scaling (true scale is unwatchable — plan for it)
- **Three display modes**, toggleable in the viewer:
  1. **True scale** — radii rendered at actual km. Planets are sub-pixel next to the Sun. Included for honesty, not usability.
  2. **Log-exaggerated** (default) — `display_radius = true_radius * k`, where `k` is chosen so Earth renders at ~1% of 1 AU. Inner and outer planets both visible.
  3. **Fixed-size markers** — every body a billboard sphere of constant screen size, with a label. Best for seeing orbital structure at any zoom level.
- **Orbit distances are always true scale.** Only body radii get exaggerated. Exaggerating distances destroys the physics you're trying to see.

#### Camera modes
- **Top-down ecliptic** (default) — camera above the ecliptic plane, inner system framed. Good for watching orbits.
- **Free orbit** — standard turntable / trackball with pan, zoom, rotate. pyvista default.
- **Follow body** — camera focal point locked to a selected body; orbit the body with the mouse. Switch target from a dropdown. Implemented via `plotter.camera.focal_point` updated each frame.
- **Rotating frame** — camera co-rotates with a selected primary pair (e.g., Sun-Earth). This is what makes Lagrange tadpoles actually visible; without it, L4 trajectories look like a messy spiral. Implemented by applying a per-frame rotation to all rendered positions around the barycenter.

#### Orbit trails
- Per-body polyline of recent positions. Two trail modes:
  - **Length-based** — last N samples (default N=500).
  - **Time-based** — last T sim-time (e.g., last 1 orbital period of the body).
- Alpha fades from head (current position) to tail. Trail color matches body color.
- Toggle per body; mute the inner system's tangled trails when you want to watch outer planets.

#### Time UI
- Slider mapped to state-history sample index. Displays current sim-epoch (ISO TDB) and elapsed-from-start.
- Play / pause / reverse, with speed multiplier (1 day/s, 30 days/s, 1 yr/s).
- Jump-to-epoch input for typing a date.
- Keyboard: space = play/pause, left/right = step, [ / ] = slower/faster.

#### Body appearance
- M1: flat-colored spheres (pyvista `Sphere`) keyed to a palette, text label sprites above each body.
- M6 polish: optional textures (earth.jpg, etc.) mapped to `pyvista.Texture` on the sphere; sun rendered with an emissive material.
- Saturn's rings in M6 only — not worth the complexity before then.

## Iterative milestones (each runnable end-to-end)

- **M0 — Bootstrap.** Prove the toolchain before any physics.
  - `git init` in `C:\git\tommybship\tomcosmos`; add `.gitignore` (see GitHub integration section) and `LICENSE` (MIT). First commit is the empty scaffolding so later `git_sha` captures in run metadata have something to point at.
  - Create `tommybship/tomcosmos` on github.com and push: `git remote add origin git@github.com:tommybship/tomcosmos.git && git push -u origin main`.
  - `.github/workflows/ci.yml` stubbed with a smoke pytest job (full matrix set up in M1 once there's real code to test).
  - Conda env built from `environment.yml` (full pinned list in the Dependency plan section above). Env name: `tomcosmos`.
  - Repo layout scaffolded (matches the tree above): `src/tomcosmos/` with subpackages, `tests/`, `scenarios/`, `data/kernels/` (gitignored), `scripts/`, `runs/` (gitignored), `environment.yml`, `pyproject.toml` (editable install + `[project.scripts]` entry for the `tomcosmos` CLI), `README.md`.
  - `scripts/fetch_kernels.py` downloads what's needed into `data/kernels/`:
    - M1: skyfield auto-downloads `de440s.bsp` (smaller subset of DE440) on first use — no manual fetch needed initially.
    - M2 onward: `naif0012.tls` (leapseconds) + `de440.bsp` (planets) + satellite kernels (`jup365.bsp`, `sat441.bsp`, etc.) from [NAIF generic kernels](https://naif.jpl.nasa.gov/pub/naif/generic_kernels/).
  - `scripts/hello_world.py` loads skyfield, prints Earth's heliocentric position at "now" — confirms ephemeris works end-to-end.
  - **REBOUND units setup** lives in `state/integrator.py`: `sim.units = ('AU', 'yr', 'Msun')` *internally* (gives `G ≈ 39.48`, numerically well-scaled for planetary dynamics). **Km / km-per-s is the I/O boundary unit** — StateHistory is stored in km / km/s, SPICE queries return km / km/s, and `state/frames.py` exposes conversion helpers on both sides. Mixing 1e27 kg masses with 1e8 km distances inside 64-bit doubles loses digits of precision in force sums; natural units avoid this. Never use REBOUND's default `G=1` — silent unit mismatches. Enforce via a helper (`build_simulation`) that creates the `Simulation`, sets units, adds bodies with stable hashes, and calls `sim.move_to_com()` in one shot.
  - **Particle identity by hash, never index.** REBOUND's `sim.remove()` reshuffles indices, so every particle is added with a deterministic `hash=<name>` (bodies) or `hash=<uuid>` (test particles). All later lookups — applying Δv events, marking terminated on impact, reading state — go through the hash. Load-bearing for M3 (variable test-particle counts), M4 (impact removal mid-run), and M5 (10k bodies where any index assumption breaks).
  - **Always `sim.move_to_com()` at t=0.** Without it, the system COM drifts linearly through ICRF and energy / angular-momentum diagnostics pick up spurious linear terms that look like integrator drift but aren't.
  - `tests/test_smoke.py` imports `rebound`, creates a 2-body Simulation with our units helper, integrates one step, asserts it didn't crash. CI-ready even if CI is just `pytest` locally.
  - Exit criterion for M0: (1) `conda env create -f environment.yml && conda activate tomcosmos && pip install -e . && python scripts/hello_world.py && pytest` all succeed on a clean checkout; (2) CI smoke workflow green on the first push to `main`.

- **M1 — Sun + 8 planets, validated.** skyfield ICs at chosen epoch → REBOUND WHFast → pyvista with orbit trails.
  - **Exit criteria:** (1) `tomcosmos run scenarios/sun-planets.yaml` produces a Parquet file; (2) `tomcosmos view runs/sun-planets-baseline.parquet` opens the 3D viewer with all 9 bodies and trails; (3) tests in Tier 1 + Tier 2 pass; (4) Earth position after 1 simulated year matches skyfield to <1000 km; (5) `tomcosmos info` shows embedded metadata (git SHA, kernel hashes, energy-error series).

- **M2 — Major moons.** Luna, Galileans, Titan, Saturnian system. Migrate to spiceypy for satellite kernels.
  - **Exit criteria:** (1) Scenario including at least Earth+Moon and Jupiter+4 Galileans runs to completion with IAS15; (2) moons visibly orbit their primaries in the viewer (use the "follow body" camera on Jupiter to see Io zip around); (3) `EphemerisSource` ABC has both `SkyfieldSource` and `SpiceSource` implementations and tests parameterize over both; (4) energy error stays ≤1e-12 over a 10-year run.

- **M3 — Test particles + Lagrange demo.** Massless particles with user-specified ICs. The "does the physics really work" milestone.
  - **Exit criteria:** (1) All three IC types (`explicit`, `lagrange`, `keplerian`) work and have unit tests; (2) `scenarios/sun-earth-l4-tadpole.yaml` runs for 50 years; (3) test particle stays within ±10° of L4's equilibrium in the Sun-Earth rotating frame — **and the rotating-frame camera shows the tadpole trajectory clearly**; (4) close-encounter logging (Hill-sphere entry/exit) fires for any particle that crosses the Hill radius of a body.

- **M4 — Spacecraft scenarios.** Named particles with optional Δv events. Pretend-mission-planning.
  - **Exit criteria:** (1) Δv events in scenario YAML are applied at the correct epoch (unit test: energy change matches `m * |dv|²/2` for a single instantaneous burn); (2) MERCURIUS runs with a probe passing through Earth's Hill sphere, no numerical blowup; (3) a Hohmann-like transfer scenario Earth→Mars completes and the probe's trajectory visibly intersects Mars's orbit within ~5° of the expected phase angle — precise targeting needs a Lambert solver / iterated midcourse correction, which is a non-goal; (4) event log in `runs/<basename>.events.parquet` records every Δv and encounter.

- **M5 — Small bodies.** Ingest from JPL Small-Body Database.
  - **Critical gotcha:** SBDB returns each body's elements at *its own* epoch (typically the body's most recent orbit-determination epoch, not the scenario epoch). Every body's elements must be Kepler-propagated from `elements.epoch` to `scenario.epoch` (two-body mean-motion advance on `M`, then re-solve) *before* converting to state vectors. Skipping this silently places bodies at positions from months-to-years off — no error, just wrong sim. `resolve_keplerian_ic` takes an `elements_epoch` parameter for exactly this.
  - **Exit criteria:** (1) `astroquery.jplsbdb` query for N asteroids returns elements that convert cleanly to our `keplerian` IC type, with per-body epoch propagation; (2) round-trip test: ingest SBDB elements for a known asteroid, propagate to `t0 + 1 year`, compare to a re-query at that date — agreement < 1000 km (pure-Keplerian; real N-body adds perturbations); (3) scenario with Sun+8 planets + 1,000 asteroids completes in <5× the performance-target time; (4) viewer renders 1,000 asteroids without dropping below interactive framerate (instanced spheres or point sprites if needed); (5) analysis helper `encounters.py` can identify asteroids that come within 0.1 AU of Earth over the run's duration.

- **M6 — Share + package.**
  - **Exit criteria:** (1) `tomcosmos serve runs/<name>.parquet` launches a trame-based web viewer on localhost with the same features as the desktop pyvista viewer; (2) README has a quickstart that works on a fresh machine (verified by following it yourself on a fresh conda env); (3) at least one textured body (Earth or Sun) to prove the asset pipeline works; (4) optional: PyInstaller bundle.

## Developer workflow + CLI

Two supported entry points that share the same core library.

### CLI (everyday use)
Installed as a console script via `pyproject.toml` `[project.scripts]`. Built with `typer` (declarative, generates `--help` automatically).

```
tomcosmos run     scenarios/sun-planets.yaml        # integrate; writes runs/<name>.parquet
tomcosmos view    runs/sun-planets-baseline.parquet # open pyvista viewer
tomcosmos serve   runs/sun-planets-baseline.parquet # trame web viewer on localhost (M6)
tomcosmos validate scenarios/sun-planets.yaml       # full preflight: schema + ephemeris coverage + kernels + body/event lookups
tomcosmos info    runs/sun-planets-baseline.parquet # print metadata (epoch, duration, energy error)
tomcosmos fetch-kernels                             # download SPICE kernels to data/kernels/
```

Design rules:
- `run` and `view` are **completely decoupled** — no shared process state. You can view any old run without re-running.
- `run` is reproducible from the scenario YAML + kernel set alone on the same machine (same BLAS/CPU). Seed any stochastic IC sampling. Byte-identical across platforms is a non-goal; agreement to ~1e-10 is the bar.
- Every command accepts `--verbose` and `--log-file`. Logs are structured (JSON) so they're greppable.
- `validate` is a *preflight*, not just schema checking. It loads kernels, confirms `epoch + duration` falls within every ephemeris source's coverage, checks that each Δv / event epoch is inside the run window, resolves every body's mass/radius from `constants.py` (catches typos in `name` / `spice_id` with a "did you mean" suggestion), and dry-runs the IC computation for test particles. Schema-only Pydantic validation happens at `Scenario.from_yaml`; `validate` does everything else that would otherwise fail mid-run.

### Python API (notebooks, experiments, scripting)
The CLI is a thin layer over the library. All of this should be equally usable from a Jupyter notebook:

```python
from tomcosmos import Scenario, run, load_run

scenario = Scenario.from_yaml("scenarios/sun-planets.yaml")
history = run(scenario)                           # returns an in-memory StateHistory
history.to_parquet("runs/custom.parquet")

# Later, or in a different process:
history = load_run("runs/custom.parquet")
history.plot_trajectory("Earth")                  # matplotlib quick-look
history.viewer().show()                           # pyvista
```

`StateHistory` exposes pandas/polars DataFrames for arbitrary analysis — energy traces, distance between bodies, encounter detection — without forcing the user through the CLI.

### `run()` orchestration flow
The core function that turns a `Scenario` into a `StateHistory` on disk. Roughly:

```python
def run(scenario: Scenario, *, allow_dirty: bool = False) -> StateHistory:
    # 1. Validate environment and capture metadata
    if not allow_dirty and _git_is_dirty():
        raise DirtyWorkingTreeError
    metadata = RunMetadata.capture(scenario)   # git_sha, kernel_hashes, versions, ...

    # 2. Resolve ephemeris source and check time range
    source = EphemerisSource.for_scenario(scenario)   # SkyfieldSource or SpiceSource
    source.require_covers(scenario.epoch, scenario.duration)

    # 3. Resolve bodies: mass/radius from constants or explicit; IC from ephemeris or explicit
    bodies = [resolve_body(b, scenario.epoch, source) for b in scenario.bodies]

    # 4. Resolve test particles: lagrange / keplerian / explicit -> (r, v)
    test_particles = [resolve_test_particle(p, scenario.epoch, bodies) for p in scenario.test_particles]

    # 5. Build REBOUND Simulation with units, integrator, particles
    sim = build_simulation(bodies, test_particles, scenario.integrator)
    # sim.units = ('km', 's', 'kg'); sim.integrator = 'whfast'; ...

    # 6. Register event callbacks (collisions, Δv events, Hill-sphere entry)
    event_log = EventLog()
    attach_callbacks(sim, scenario, event_log)

    # 7. Open output writer (streaming to Parquet)
    writer = StateHistoryWriter(scenario.output.path, bodies, metadata)

    # 8. Integration loop with output cadence
    for t in scenario.output_times():
        sim.integrate(t)
        check_energy_error(sim, threshold=scenario.integrator.divergence_threshold)
        writer.append_sample(sim, event_log.drain())
        maybe_checkpoint(sim, scenario, t)

    # 9. Finalize: flush writer, write event-log Parquet, close log handlers
    writer.close()
    event_log.to_parquet(scenario.output.events_path)
    return StateHistory.from_parquet(scenario.output.path)
```

This is a sketch; actual implementation splits into `state/integrator.py` (steps 5–6, 8), `io/history.py` (step 7), `io/diagnostics.py` (step 1). The CLI's `tomcosmos run` is a thin wrapper calling `run()`.

## Testing strategy

Three tiers, all under `pytest`. Run `pytest -n auto` for parallelism (via `pytest-xdist`).

### Tier 1 — unit (fast, always run)
- **Frame conversions** round-trip: `icrf → ecliptic → icrf` is identity to machine precision.
- **Unit handling**: `parse_duration("10 yr")` → `10 * u.year`; reject bad inputs.
- **Scenario schema**: Pydantic validation accepts known-good fixtures, rejects known-bad ones (missing SPICE id, conflicting `source` and explicit state, etc.).
- **Lagrange IC math** (see below): L4 position for Sun-Earth matches closed-form to 1 km.
- **Kepler-equation solver** round-trip: orbital elements → state → elements reproduces inputs.

### Tier 2 — physics invariants (medium, marked `@pytest.mark.physics`)
- **Energy conservation (WHFast)**: integrate Sun+Jupiter for 1000 years, 1-day step. Assert `|ΔE/E|` bounded below 1e-10 and non-drifting (linear regression slope near zero).
- **Angular momentum**: same setup, assert `|ΔL/L|` bounded below 1e-12.
- **Kepler's third law**: for each planet in a baseline run, measure orbital period from zero-crossings; compare to `(a^3 / M_sun)^{1/2}` with a tolerance.
- **Lagrange stability (M3)**: test particle at L4 for 10 years, assert it stays within 10° of L4's equilibrium angle in the Sun-Earth rotating frame (tadpole bound, not escape).

### Tier 3 — ephemeris agreement (slow, marked `@pytest.mark.ephemeris`)
- Propagate REBOUND from ephemeris ICs; compare to ephemeris lookup at future epochs.
- Tolerance envelope degrades with time (physics, not bug):
  - 1 year: <1000 km for Earth, <10,000 km for Mercury.
  - 10 years: <100,000 km for Earth.
  - 100 years: <10,000,000 km for Earth (loose — no GR, no tides, no asteroid-belt perturbations).
- Run weekly or on release, not on every commit.

### Viewer snapshot tests
- Render a known scenario, compare output image to a committed baseline using `pytest-mpl` or `pyvista.Plotter.screenshot` + image diff. Catches regressions in the viz code without requiring humans to eyeball everything. Marked `@pytest.mark.viewer`, skippable on headless CI.

### Fixtures
- `tests/fixtures/scenarios/` — known-good and known-bad YAMLs.
- `tests/fixtures/expected/` — expected state vectors (small CSVs) for canonical runs.
- Ephemeris/kernel dependence: tests either mock `ephemeris.query()` or use a tiny bundled ephemeris subset (`de440s.bsp`, ~32 MB, commits cleanly).

## Special IC computation

Scenarios reference non-state-vector ICs. Each type is a pure function that, given the `epoch` and existing bodies, returns `(r, v)` in ICRF barycentric km / km/s.

### `type: lagrange`
Parameters: `primary` (body name), `secondary` (body name), `point` ∈ {L1..L5}, optional `offset` (small km displacement to place the particle near, not exactly at, the point).

Computation:
1. Pull primary's and secondary's state vectors at `epoch` from the already-resolved bodies list.
2. Compute instantaneous Hill-problem geometry:
   - **Collinear points (L1, L2, L3)**: solve Euler's quintic for mass ratio `μ = m_secondary / (m_primary + m_secondary)` using `scipy.optimize.brentq` on the known quintic. Closed-form to machine precision.
   - **Triangular points (L4, L5)**: 60° ahead/behind secondary on its orbit around primary, at the secondary's distance. Simple rotation in the primary-secondary orbital plane.
3. Add orbital velocity (L-points co-rotate with the primary-secondary pair: `v_particle = ω × r_particle` where `ω` is the secondary's instantaneous orbital angular velocity around the primary).
4. Transform back to ICRF barycentric and return.

Tests: analytical L4 for Sun-Earth (1 AU, 60° ahead) reproduces canonical values. L1/L2 distances for Sun-Earth match published (~1.5M km).

### `type: keplerian`
Parameters: `central_body`, classical orbital elements (`a`, `e`, `i`, `Ω`, `ω`, `M` or `ν`), `elements_epoch` (optional; defaults to scenario epoch), `frame` (optional; default `ecliptic_j2000`).

**Epoch of elements vs scenario epoch.** If `elements_epoch != scenario.epoch`, the elements are Kepler-propagated forward (or back) by `Δt = scenario.epoch - elements_epoch` using the central body's mean motion `n = sqrt(GM/a³)` applied to `M` (`M' = M + n*Δt`, renormalize modulo 2π, re-solve Kepler's equation). M5 SBDB ingestion depends on this; M3/M4 hand-written scenarios typically omit `elements_epoch` and get it for free.

**Frame ambiguity matters.** Orbital elements in the **ecliptic J2000** frame differ from elements in the **ICRF/equatorial J2000** frame by the ~23.4° obliquity rotation. Neglecting this is a common silent error — a probe launched with "ecliptic" elements interpreted as equatorial will miss Mars by millions of km.

Supported `frame` values:
- `ecliptic_j2000` (default) — standard for heliocentric orbits. Matches JPL's published Keplerian elements for most bodies.
- `icrf` — elements already in ICRF/equatorial. Use if you're importing from an SPK file or ICRF-native tool.
- `body_equator` — elements in the central body's equatorial frame (useful for moon orbits around their primary). Requires spin-axis data from `constants.py`.

Computation:
1. Solve Kepler's equation for `E` from `M` using Newton's method (seed with Danby's formula for fast convergence; iterate to 1e-14).
2. Convert `E` → `ν` → perifocal `(r, v)`.
3. Rotate through `ω`, `i`, `Ω` into the element frame (`ecliptic_j2000` by default).
4. Transform element frame → ICRF via the documented fixed rotation (obliquity for ecliptic; body-spin for `body_equator`).
5. Translate to ICRF barycentric using `central_body`'s state at `epoch`.

Tests: round-trip `elements → state → elements` for 100 random elliptical orbits in each supported frame; eccentricities up to 0.95 should round-trip to 1e-10. Cross-frame test: elements specified in ecliptic vs. ICRF frames produce states differing by exactly the obliquity rotation (sanity check that the frame handling actually fires).

### `type: explicit`
Parameters: `r` (km, 3-vector), `v` (km/s, 3-vector), optional `frame` (default: `icrf_barycentric`).

Supported `frame` values: `icrf_barycentric` (default), `icrf_heliocentric`, `ecliptic_j2000_barycentric`, `ecliptic_j2000_heliocentric`, plus `body_centric:<body_name>` for state relative to a named body.

Computation: frame conversion via `state/frames.py` helpers; otherwise pass-through. Trivial but worth having as a distinct type — it's the escape hatch for manual ICs.

### Spacecraft delta-v events (M4)
Extension to scenario YAML (applied during integration, not at IC):

```yaml
test_particles:
  - name: hohmann-probe
    ic: { type: keplerian, central_body: Sun, a: 1.26, e: 0.207, ... }
    events:
      - { type: delta_v, at: "2026-07-01T00:00 TDB", dv: [0, 2.943, 0] }  # km/s, body-frame
```

Implementation: REBOUND has `sim.integrate(tmax)` in a loop that checks for upcoming events; at each event, pause, apply `Δv` to the named particle, resume. Event log persisted into run metadata.

## Reproducibility + diagnostics

### Run metadata (embedded in every Parquet output)
Parquet supports key/value metadata on the file and per column. Every run embeds:
- `scenario_sha256` — hash of the normalized scenario YAML (post-Pydantic validation).
- `scenario_yaml` — the full scenario as a string (so you can re-run even if the file is deleted).
- `git_sha` — commit of the `tomcosmos` package at run time (captured via `git rev-parse HEAD`; fail the run if the working tree is dirty and `--allow-dirty` isn't passed).
- `rebound_version`, `astropy_version`, `python_version`, `platform`.
- `kernel_hashes` — SHA256 of each SPICE kernel used (catches silent kernel swaps).
- `start_wallclock`, `end_wallclock`, `wallclock_seconds`.
- (energy-error series is *not* embedded as metadata — it lives in the `energy_rel_err` column, see StateHistory schema above.)

`tomcosmos info <run.parquet>` prints all of the above.
`tomcosmos rerun <run.parquet>` extracts the embedded scenario and runs it again — useful for reproducing someone else's result without the original YAML in hand.

### Logging
- `structlog` for structured JSON logs.
- Every sim run logs at minimum: start (scenario name, epoch, duration), integrator setup, checkpoints every `output_cadence` with `|ΔE/E|` and wall-clock rate, end (success/fail, summary).
- Default: log to `runs/<name>.log` alongside the Parquet. `--log-file -` for stderr.

### Determinism rules
- No wall-clock seeding — any randomness (e.g., sampling small-body populations in M5) takes an explicit seed from the scenario YAML.
- REBOUND with fixed timestep is deterministic on a given machine/BLAS. Document that cross-platform byte-identical output is **not** a goal (floating-point associativity makes it intractable); instead, numerical agreement to ~1e-10 is the standard.

## Critical files (rough layout when building)
- `scenarios/*.yaml` — scenario configs
- `src/tomcosmos/cli.py` — typer CLI (`run`, `view`, `serve`, `validate`, `info`, `fetch-kernels`, `rerun`)
- `src/tomcosmos/state/scenario.py` — Pydantic scenario schema
- `src/tomcosmos/state/ephemeris.py` — SPICE/skyfield loading, state queries
- `src/tomcosmos/state/ic.py` — special IC computation (Lagrange, Keplerian, explicit)
- `src/tomcosmos/state/integrator.py` — REBOUND wrapper: `Scenario` → `Simulation`
- `src/tomcosmos/state/frames.py` — frame conversions, astropy-backed
- `src/tomcosmos/io/history.py` — Parquet state history I/O with embedded metadata (HDF5 fallback if streaming writes needed later)
- `src/tomcosmos/io/diagnostics.py` — structlog setup, run-metadata capture (git SHA, kernel hashes)
- `src/tomcosmos/viz/pyvista_viewer.py` — 3D viewer
- `src/tomcosmos/viz/web.py` — trame web app (M6)
- `scripts/fetch_kernels.py` — kernel downloader (also exposed via CLI)
- `scripts/hello_world.py` — M0 toolchain-sanity script
- `.github/workflows/ci.yml` — pytest + ruff + mypy on push/PR
- `.github/workflows/weekly.yml` — slow ephemeris-agreement tests on cron
- `tests/test_scenario_schema.py`, `tests/test_ic_lagrange.py`, `tests/test_physics_invariants.py`, `tests/test_ephemeris_agreement.py`, `tests/test_viewer_snapshot.py` — mirror the testing tiers

## Reuse — do not build these yourself
(See the Dependency plan section for the full list with version pins; this is the physics-library shortlist.)
- **`REBOUND`** — N-body integration. Do not roll your own integrator. WHFast/IAS15/MERCURIUS are world-class.
- **`skyfield`** / **`spiceypy`** — ephemeris access. Do not parse DE/SPK kernels yourself.
- **`astropy`** — time scales, units, constants, coordinate frames. Don't reimplement TDB↔UTC or hand-roll unit handling.
- **`scipy.optimize.brentq`** and Newton iteration — for Lagrange quintic and Kepler's equation. Don't write your own root finders.
- **`pyvista`** — VTK wrapper. Don't touch VTK directly.
- **`astroquery.jplsbdb`** — JPL Small-Body Database queries (M5). Don't scrape the HTML tool.

## Verification
- **M1 accuracy**: tests asserting REBOUND-propagated positions match ephemeris within tolerance over 1-year, 10-year, 100-year spans (allow tolerance to grow — you're not modeling all of JPL's physics, so real N-body *should* drift from reality).
- **Lagrange (M3)**: integrate a test particle at L4 for ≥10 years; assert it stays bounded around L4 (tadpole/horseshoe, not escape).
- **Energy conservation**: with WHFast, log relative energy error per step; should be bounded (symplectic property), not drifting linearly — if it drifts, the integrator or timestep is wrong.
- **Eyeball**: load a scenario, run sim, open viewer, scrub time, confirm orbits close at the expected periods and planets don't visibly crash into each other.

## Operational behaviors

Design decisions that affect code structure and deserve to be pinned now.

### Close-encounter semantics (test particles near massive bodies)
Applies to M3+. A test particle that enters a planet's Hill sphere could be anything from a valid flyby to an impact. Behavior:
- **Default: log and continue.** Record each Hill-sphere entry/exit in the run's event log (body, particle, time, closest approach distance, relative velocity). Particle keeps integrating.
- **Impact criterion**: `distance < body.radius_km`. On impact, the particle is marked "terminated" in the state history (NaN for subsequent samples), event logged, sim continues without it. Never silently delete — preserving the trail is important for analysis.
- **Opt-in stricter behavior** per scenario: `events: { on_impact: halt }` stops the sim entirely; `on_impact: reject_scenario` (pre-run) disallows IC that would impact within the duration.
- Implementation: REBOUND's `sim.collision = "direct"` + a custom `collision_resolve` callback that writes to the event log and either removes or halts.

### Integrator divergence detection
- On every output cadence, check `|ΔE/E|` against a scenario-configured threshold. **Default by integrator:** WHFast `1e-8`, IAS15 `1e-10`, MERCURIUS `1e-8`. `scenario.integrator.divergence_threshold` overrides. Per-integrator defaults because IAS15's per-step target is machine-precision — a 1e-8 bar on IAS15 would miss real blowup.
- On breach: log ERROR with the current state, abort the run, write partial Parquet with `status=diverged` in metadata. Don't silently finish a corrupted run.

### Checkpoint / resume
- For runs over ~1 hour wall-clock (expected at M2's long integrations and M5's many particles), REBOUND's `sim.save_to_file(path)` snapshots the full simulation state every `checkpoint_interval` (default: every 100 output samples).
- Checkpoint lives alongside the partial Parquet: `runs/<name>.checkpoint`.
- `tomcosmos run --resume <name>` loads the most recent checkpoint, reads the already-written Parquet samples, and continues. Idempotent: resuming a completed run is a no-op.
- Disabled by default for short runs (< 100 samples); scenarios toggle via `output: { checkpoint: true }`.

### Performance targets (order-of-magnitude, not contracts)
Concrete numbers per milestone so "too slow" has a meaning:

| Scenario | Duration | Integrator | Expected wall-clock | Notes |
|---|---|---|---|---|
| M1 Sun+8 planets | 10 yr | WHFast, 1-day step | **~2–5 s** | ~3,650 steps × 9 bodies; REBOUND is fast |
| M1 Sun+8 planets | 1,000 yr | WHFast, 1-day step | **~3–8 min** | Baseline long run |
| M2 + major moons (~15 bodies) | 10 yr | IAS15 adaptive | **~30–90 s** | Adaptive step shrinks around Io |
| M3 Sun-Earth + 1 L4 particle | 100 yr | IAS15 | **~10–30 s** | Small N, very accurate |
| M5 Sun+planets + 10k asteroids | 10 yr | WHFast, `testparticle_type=0` | **~5–15 min** | Test particles don't cross-interact |

If you see 10× these numbers, something is wrong (wrong integrator, wrong units, BLAS misconfigured). Don't chase micro-optimizations until you're within an order of magnitude.

### Error handling philosophy
- **Input errors → fail at load**: Pydantic catches scenario issues before any integration starts. User sees the full validation report, not a cryptic crash 5 minutes into a run.
- **Physics errors → fail loudly with context**: divergence, missing kernel body, bad frame transform → raise `tomcosmos.exceptions.*` with enough info to diagnose (current sim time, offending body, last valid state). Never raise bare `ValueError`.
- **External-resource errors → retry with backoff**: JPL SBDB API calls (M5), kernel downloads (M0 setup). Three attempts, exponential backoff, then fail.
- **No silent fallbacks.** If scenario asks for MERCURIUS and it's unavailable, error out; don't silently swap to IAS15.

## Open questions for later (not blocking)
- **Rotations/spin axes** — deferred; add when first needed (ground track on a planet, tidally-locked moon viz). `constants.py` has a `spin` slot ready.
- **Relativistic corrections** — deferred. Probably never needed here, but REBOUND has `REBOUNDx` add-ons (`gr`, `gr_potential`) if you want to experiment with Mercury's perihelion precession.
- **Saturn's rings, planetary textures, and eclipse shadowing** — visual polish; M6 if ever.
- **Alternate distribution** — PyInstaller bundle for desktop, Docker image for web viewer. Defer until someone actually asks for a packaged build.
- **Time precision beyond astropy defaults** — astropy's `Time` uses two-double internally (jd1 + jd2) to keep sub-millisecond precision over millennia. If you need better, revisit; unlikely.
- **Barycenter-drift between skyfield and SPICE** — the Sun's position relative to the SSB differs slightly between ephemeris sources. Document the tolerance envelope in the M1→M2 migration so tests don't get spuriously brittle.
- **Scenario inheritance / composition** — YAML anchors or `extends: other-scenario.yaml`. YAGNI until scenarios duplicate themselves.
- **`fetch-kernels` caching** — re-download policy, checksum verification. Simple "download if missing" is fine until it isn't.
- **Pre-commit hooks** — ruff + mypy on commit. Nice, not needed.
- **Log rotation** — long runs produce large logs. One-file-per-run with manual cleanup is fine until `runs/` becomes unwieldy.
- **Rendering performance** — pyvista with 10,000 small bodies at 60fps is not guaranteed; M5 may need instanced spheres or point sprites. Profile when you get there.

## Glossary

Jargon that appears throughout the plan, compressed.

- **AU** — Astronomical Unit. ~150 million km, Earth's mean orbital distance.
- **Barycenter** — center of mass. "SSB" = Solar System Barycenter, the origin of the ICRF frame for solar system dynamics. Usually near but not at the Sun's center (Jupiter pulls it around).
- **Ecliptic** — the plane of Earth's orbit around the Sun. Most solar system orbits are close to (but not in) this plane. Tilted ~23.4° from Earth's equator.
- **Ephemeris** — tabulated positions of celestial bodies vs time. JPL publishes these as "DE" (Development Ephemeris) files: DE440, DE441, etc.
- **Energy error (`|ΔE/E|`)** — relative change in total energy from step to step. For a symplectic integrator, this should oscillate around zero with bounded amplitude — never drift linearly.
- **GR** — General Relativity. Contributes small corrections (e.g., Mercury's perihelion precession at ~43 arcsec/century). Deferred.
- **Hill sphere** — region around a body where its gravity dominates over its primary's. `r_hill ≈ a * (m / 3M)^(1/3)`. The Moon is inside Earth's Hill sphere; a comet inside Jupiter's gets slung around.
- **IAS15** — REBOUND's 15th-order adaptive Gauss-Radau integrator. High accuracy, adaptive step. Not symplectic, but energy error stays at machine precision.
- **ICRF** — International Celestial Reference Frame. The modern inertial frame for astronomy, defined by distant radio sources. Our internal sim frame.
- **Integrator** — numerical method that advances particle states by one timestep. Examples: leapfrog, Runge-Kutta, symplectic Wisdom-Holman.
- **Jacobi coordinates** — coordinate system where each body's position is relative to the center of mass of all interior bodies. Useful for Wisdom-Holman integrators.
- **Keplerian orbit** — two-body closed elliptical orbit described by 6 orbital elements. Doesn't account for third-body perturbations.
- **Lagrange points (L1–L5)** — five positions in a two-body system where a small mass can stay stationary relative to the two primaries. L1–L3 collinear and unstable; L4/L5 triangular and stable for mass ratios < 1/25 (Sun-Earth qualifies).
- **Libration** — oscillation around an equilibrium. A tadpole orbit librates around L4.
- **MERCURIUS** — REBOUND's hybrid integrator: Wisdom-Holman far from encounters, IAS15 inside Hill spheres. Good for spacecraft.
- **N-body** — simulation where every particle gravitationally affects every other. O(N²) naively; can be O(N) for test particles that don't affect massive bodies.
- **NAIF / SPICE** — NASA's toolkit and file formats for ephemeris, attitude, and mission data. `spiceypy` is the Python binding.
- **Osculating orbital elements** — at any instant, the 6 Keplerian elements of the two-body orbit the particle is currently on, ignoring perturbations. Changes over time as perturbations act.
- **Patched conics** — mission-design approximation: treat trajectory as sequential two-body orbits, each patched at the sphere-of-influence boundary. Cheap but loses Lagrange-point dynamics.
- **PCK** — Planetary Constants Kernel. SPICE file type containing masses, radii, spin rates for solar system bodies.
- **Perihelion / aphelion** — closest / farthest point on an elliptical orbit from the Sun.
- **Perturbation** — deviation from a pure two-body orbit caused by other bodies, oblateness, radiation pressure, etc.
- **SPK** — SPICE Position Kernel. Binary file of Chebyshev coefficients for body positions over time. DE440 is an SPK.
- **SSB** — Solar System Barycenter. Origin for ICRF-barycentric states.
- **Symplectic** — property of integrators that preserve the Hamiltonian structure of a conservative system. Energy doesn't drift; phase-space volume is preserved. WHFast is symplectic.
- **Tadpole orbit** — closed trajectory that librates around L4 or L5, shaped like a tadpole in the rotating frame.
- **TDB / TT / UTC** — time scales. TDB (Barycentric Dynamical Time) for solar system dynamics, TT (Terrestrial Time) for Earth-local, UTC (Coordinated Universal Time) for clocks. Differ by leap seconds (UTC) and relativistic corrections (TDB vs TT).
- **Test particle** — a particle with zero mass, affected by gravity but not producing any. Cheap to simulate in large numbers.
- **WHFast** — REBOUND's Wisdom-Holman fast symplectic integrator. Fixed timestep, excellent for long-term planetary dynamics.
