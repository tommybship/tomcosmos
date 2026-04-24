# tomcosmos

A Python solar system state simulator: accurate N-body integration, ephemeris-sourced initial conditions, 3D visualization.

## What it does

- Integrates the solar system (Sun + planets, optionally moons and test particles) from real ephemeris initial conditions using REBOUND.
- Runs scenarios defined in YAML; writes state histories to Parquet with full reproducibility metadata.
- Renders results in an interactive 3D viewer (pyvista desktop; trame web later).

## Status

**M0 — bootstrap.** Scaffolding only; no physics yet. See `.claude/plans/if-i-wanted-to-lively-river.md` (outside this repo) for the full roadmap.

## Quickstart

```bash
conda env create -f environment.yml
conda activate tomcosmos
pip install -e .
python scripts/hello_world.py
pytest
```

## Scenarios

YAML files under `scenarios/` describe what to simulate. See the plan for schema; examples appear as milestones land.

## Accuracy notes

Pure N-body from ephemeris ICs, not a replacement for JPL DE. Expect drift from real-world positions over long runs — we don't model GR, asteroid-belt perturbations, or tides. Use the ephemeris lookup as ground truth.

## Development

- Conda env: `environment.yml` (conda-forge)
- Tests: `pytest -n auto`
- Lint/format: `ruff check . && ruff format .`
- Types: `mypy src/tomcosmos/state`

## License

MIT — see `LICENSE`.
