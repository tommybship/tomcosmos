import os
import tempfile
from pathlib import Path
from typing import Any

import pytest
import yaml
from astropy import units as u
from astropy.time import Time
from pydantic import ValidationError

from tomcosmos import Scenario, ScenarioValidationError

FIXTURES = Path(__file__).parent / "fixtures" / "scenarios"


def _good_dict() -> dict[str, Any]:
    """Minimal valid scenario for dict-level validation tests."""
    return {
        "schema_version": 1,
        "name": "test",
        "epoch": "2026-04-23T00:00:00 TDB",
        "duration": "1 yr",
        "integrator": {"name": "whfast", "timestep": "1 day"},
        "output": {"format": "parquet", "cadence": "1 day"},
        "bodies": [
            {"name": "sun", "spice_id": 10, "ic": {"source": "ephemeris"}},
            {"name": "earth", "spice_id": 399, "ic": {"source": "ephemeris"}},
        ],
    }


def _load_dict_via_yaml(d: dict[str, Any], tmp_path: Path | None = None) -> Scenario:
    """Helper: write dict to a temp YAML and load via Scenario.from_yaml."""
    if tmp_path is None:
        fd, p = tempfile.mkstemp(suffix=".yaml")
        os.close(fd)
        path = Path(p)
    else:
        path = tmp_path / "scenario.yaml"
    path.write_text(yaml.safe_dump(d), encoding="utf-8")
    return Scenario.from_yaml(path)


def test_good_scenario_from_yaml() -> None:
    s = Scenario.from_yaml(FIXTURES / "good_sun_planets.yaml")
    assert s.name == "sun-planets-baseline"
    assert isinstance(s.epoch, Time)
    assert s.epoch.scale == "tdb"
    assert s.duration.to(u.yr).value == pytest.approx(10.0)
    assert s.integrator.name == "whfast"
    assert len(s.bodies) == 9
    assert s.test_particles == []


def test_good_dict_validates() -> None:
    s = Scenario.model_validate(_good_dict())
    assert s.integrator.timestep is not None
    assert s.integrator.timestep.to(u.s).value == pytest.approx(86400.0)


def test_missing_schema_version_rejected() -> None:
    d = _good_dict()
    del d["schema_version"]
    with pytest.raises(ScenarioValidationError, match="schema_version"):
        _load_dict_via_yaml(d)


def test_unknown_schema_version_rejected(tmp_path: Path) -> None:
    d = _good_dict()
    d["schema_version"] = 999
    with pytest.raises(ScenarioValidationError, match="not supported"):
        _load_dict_via_yaml(d, tmp_path)


def test_duplicate_body_names_rejected() -> None:
    d = _good_dict()
    d["bodies"].append(
        {"name": "earth", "spice_id": 399, "ic": {"source": "ephemeris"}}
    )
    with pytest.raises(ValidationError, match="duplicate"):
        Scenario.model_validate(d)


def test_whfast_without_timestep_rejected() -> None:
    d = _good_dict()
    d["integrator"] = {"name": "whfast"}
    with pytest.raises(ValidationError, match="timestep"):
        Scenario.model_validate(d)


def test_ias15_with_timestep_rejected() -> None:
    d = _good_dict()
    d["integrator"] = {"name": "ias15", "timestep": "1 day"}
    with pytest.raises(ValidationError, match="adaptive"):
        Scenario.model_validate(d)


def test_ias15_without_timestep_ok() -> None:
    d = _good_dict()
    d["integrator"] = {"name": "ias15"}
    s = Scenario.model_validate(d)
    assert s.integrator.timestep is None


def test_unknown_integrator_rejected() -> None:
    d = _good_dict()
    d["integrator"] = {"name": "leapfrog", "timestep": "1 day"}
    with pytest.raises(ValidationError):
        Scenario.model_validate(d)


def test_negative_duration_rejected() -> None:
    d = _good_dict()
    d["duration"] = "-5 yr"
    with pytest.raises(ValidationError, match="positive"):
        Scenario.model_validate(d)


def test_zero_duration_rejected() -> None:
    d = _good_dict()
    d["duration"] = "0 s"
    with pytest.raises(ValidationError, match="positive"):
        Scenario.model_validate(d)


def test_non_time_duration_unit_rejected() -> None:
    d = _good_dict()
    d["duration"] = "10 km"
    with pytest.raises(ValidationError, match="time-equivalent"):
        Scenario.model_validate(d)


def test_epoch_without_scale_rejected() -> None:
    d = _good_dict()
    d["epoch"] = "2026-04-23T00:00:00"
    with pytest.raises(ValidationError, match="explicit scale"):
        Scenario.model_validate(d)


def test_epoch_unknown_scale_rejected() -> None:
    d = _good_dict()
    d["epoch"] = "2026-04-23T00:00:00 ZULU"
    with pytest.raises(ValidationError, match="time scale"):
        Scenario.model_validate(d)


def test_epoch_j2000_alias() -> None:
    d = _good_dict()
    d["epoch"] = "J2000"
    s = Scenario.model_validate(d)
    assert s.epoch.scale == "tdb"


def test_negative_mass_rejected() -> None:
    d = _good_dict()
    d["bodies"][0]["mass_kg"] = -1.0
    with pytest.raises(ValidationError):
        Scenario.model_validate(d)


def test_explicit_ic_body() -> None:
    d = _good_dict()
    d["bodies"][0] = {
        "name": "rogue",
        "ic": {
            "source": "explicit",
            "r": [0.0, 0.0, 0.0],
            "v": [0.0, 0.0, 0.0],
        },
    }
    s = Scenario.model_validate(d)
    assert s.bodies[0].ic.source == "explicit"


def test_explicit_ic_wrong_vector_length_rejected() -> None:
    d = _good_dict()
    d["bodies"][0] = {
        "name": "rogue",
        "ic": {
            "source": "explicit",
            "r": [0.0, 0.0],
            "v": [0.0, 0.0, 0.0],
        },
    }
    with pytest.raises(ValidationError):
        Scenario.model_validate(d)


def test_unknown_frame_rejected() -> None:
    d = _good_dict()
    d["bodies"][0] = {
        "name": "rogue",
        "ic": {
            "source": "explicit",
            "r": [0.0, 0.0, 0.0],
            "v": [0.0, 0.0, 0.0],
            "frame": "galactic_bogus",
        },
    }
    with pytest.raises(ValidationError):
        Scenario.model_validate(d)


def test_extra_fields_forbidden() -> None:
    d = _good_dict()
    d["surprise"] = "hi"
    with pytest.raises(ValidationError):
        Scenario.model_validate(d)


def test_no_bodies_rejected() -> None:
    d = _good_dict()
    d["bodies"] = []
    with pytest.raises(ValidationError):
        Scenario.model_validate(d)


def test_test_particle_with_same_name_as_body_rejected() -> None:
    d = _good_dict()
    d["test_particles"] = [
        {
            "name": "earth",
            "ic": {"type": "explicit", "r": [0, 0, 0], "v": [0, 0, 0]},
        }
    ]
    with pytest.raises(ValidationError, match="duplicate"):
        Scenario.model_validate(d)
