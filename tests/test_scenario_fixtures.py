"""Smoke-load every YAML in scenarios/ to catch silent rot.

Each committed scenario must parse, schema-validate, and (if its
required ephemeris kernel is present) survive IC resolution. The kernel-
present check matters because some scenarios deliberately need
satellite kernels that aren't part of the default install — those are
auto-skipped here when the kernel isn't on disk.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tomcosmos import Scenario
from tomcosmos.config import kernel_dir
from tomcosmos.kernels import group_for_body

SCENARIOS_DIR = Path(__file__).parent.parent / "scenarios"
SCENARIO_FILES = sorted(SCENARIOS_DIR.glob("*.yaml"))


@pytest.mark.parametrize("scenario_path", SCENARIO_FILES, ids=lambda p: p.name)
def test_committed_scenario_loads(scenario_path: Path) -> None:
    """Every scenario in scenarios/ must parse and pass schema validation.

    This is the cheap structural check; ephemeris/IC resolution lives in
    a separate test that gates on kernel availability. Mode A scenarios
    (`integrator.ephemeris_perturbers=True`) have `bodies: []` by
    construction — the schema validator already rejects mismatches
    between mode and body count, so we don't re-assert here.
    """
    scenario = Scenario.from_yaml(scenario_path)
    assert scenario.schema_version == 1
    assert scenario.name
    if scenario.integrator.ephemeris_perturbers:
        assert scenario.bodies == []
        assert scenario.test_particles, (
            "Mode A scenario must declare at least one test particle"
        )
    else:
        assert scenario.bodies


@pytest.mark.ephemeris
@pytest.mark.parametrize("scenario_path", SCENARIO_FILES, ids=lambda p: p.name)
def test_committed_scenario_resolves_ics(scenario_path: Path) -> None:
    """For every committed scenario whose required kernels are on disk,
    full IC resolution should succeed — no missing bodies, no
    out-of-range epochs, no silently-broken kernel keys."""
    from tomcosmos.state.ephemeris import EphemerisSource
    from tomcosmos.state.ic import resolve_scenario

    scenario = Scenario.from_yaml(scenario_path)

    # Skip if the scenario needs a satellite kernel that isn't fetched.
    kdir = kernel_dir()
    for body in scenario.bodies:
        group = group_for_body(body.name)
        if group is None:
            continue
        if not (kdir / group.filename).exists():
            pytest.skip(
                f"{scenario_path.name} needs {group.filename} "
                f"(run: tomcosmos fetch-kernels --include {group.name})"
            )

    source = EphemerisSource()
    source.require_covers(scenario.epoch, scenario.duration)
    bodies, particles = resolve_scenario(scenario, source)
    assert len(bodies) == len(scenario.bodies)
    assert len(particles) == len(scenario.test_particles)
