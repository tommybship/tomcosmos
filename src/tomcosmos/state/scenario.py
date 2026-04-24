"""Scenario schema — Pydantic models for declarative simulation configs.

Load via `Scenario.from_yaml(path)`. Raises `ScenarioValidationError` on
schema or semantic failures. See PLAN.md > "Scenario layer" for design.

What lives here (schema-level only):
- Structural validation (types, required fields, enums).
- Trivially checkable invariants (unique body names, positive durations,
  timestep presence matches integrator kind).

What lives in preflight (NOT here):
- Ephemeris coverage, kernel presence, SPICE ID existence, event epochs
  inside duration window. Those need loaded kernels or ephemeris data
  and are the job of `tomcosmos validate`.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from astropy import units as u
from astropy.time import Time
from astropy.units import Quantity
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PlainSerializer,
    ValidationError,
    model_validator,
)

from tomcosmos.exceptions import ScenarioValidationError

SCHEMA_VERSION = 1

_ALLOWED_SCALES = frozenset({"tdb", "tt", "utc", "tai", "tcb", "tcg"})
_DURATION_RE = re.compile(
    r"^\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s*([a-zA-Z]+)\s*$"
)
_SUPPORTED_FRAMES = (
    "icrf_barycentric",
    "icrf_heliocentric",
    "ecliptic_j2000_barycentric",
    "ecliptic_j2000_heliocentric",
)
Frame = Literal[
    "icrf_barycentric",
    "icrf_heliocentric",
    "ecliptic_j2000_barycentric",
    "ecliptic_j2000_heliocentric",
]


def _parse_epoch(v: object) -> Time:
    if isinstance(v, Time):
        return v
    if not isinstance(v, str):
        raise TypeError(
            f"epoch must be str or astropy.time.Time, got {type(v).__name__}"
        )
    s = v.strip()
    if s.upper() == "J2000":
        return Time("2000-01-01T12:00:00", scale="tdb")
    parts = s.rsplit(" ", 1)
    if len(parts) != 2:
        raise ValueError(
            f"epoch must include an explicit scale (e.g. '2026-04-23T00:00:00 TDB'); got {v!r}"
        )
    iso, scale = parts
    if scale.lower() not in _ALLOWED_SCALES:
        raise ValueError(
            f"unknown time scale {scale!r}; expected one of {sorted(_ALLOWED_SCALES)}"
        )
    return Time(iso, scale=scale.lower())


def _format_epoch(t: Time) -> str:
    return f"{t.isot} {t.scale.upper()}"


Epoch = Annotated[
    Time,
    BeforeValidator(_parse_epoch),
    PlainSerializer(_format_epoch, return_type=str),
]


def _parse_duration(v: object) -> Quantity:
    if isinstance(v, Quantity):
        q = v
    elif isinstance(v, str):
        m = _DURATION_RE.match(v)
        if not m:
            raise ValueError(
                f"duration must be '<number> <unit>' (e.g. '10 yr'); got {v!r}"
            )
        value_str, unit_str = m.groups()
        try:
            q = float(value_str) * u.Unit(unit_str)
        except ValueError as e:
            raise ValueError(
                f"unknown unit {unit_str!r} in duration {v!r}"
            ) from e
    else:
        raise TypeError(
            f"duration must be str or Quantity, got {type(v).__name__}"
        )
    if not q.unit.is_equivalent(u.s):
        raise ValueError(
            f"duration unit must be time-equivalent, got {q.unit}"
        )
    if q.value <= 0:
        raise ValueError(f"duration must be positive, got {q}")
    return q.to(u.s)


def _format_duration(q: Quantity) -> str:
    return f"{q.to(u.s).value} s"


Duration = Annotated[
    Quantity,
    BeforeValidator(_parse_duration),
    PlainSerializer(_format_duration, return_type=str),
]


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
        frozen=True,
    )


class EphemerisIc(_StrictModel):
    source: Literal["ephemeris"]


class ExplicitIc(_StrictModel):
    source: Literal["explicit"]
    r: tuple[float, float, float]
    v: tuple[float, float, float]
    frame: Frame = "icrf_barycentric"


BodyIc = Annotated[
    EphemerisIc | ExplicitIc,
    Field(discriminator="source"),
]


class Body(_StrictModel):
    name: str = Field(min_length=1)
    spice_id: int | None = None
    mass_kg: float | None = Field(default=None, gt=0)
    radius_km: float | None = Field(default=None, gt=0)
    ic: BodyIc


class TestParticleExplicitIc(_StrictModel):
    type: Literal["explicit"]
    r: tuple[float, float, float]
    v: tuple[float, float, float]
    frame: Frame = "icrf_barycentric"


# M3 will extend this union with LagrangeIc and KeplerianIc.
TestParticleIc = Annotated[
    TestParticleExplicitIc,
    Field(discriminator="type"),
]


class TestParticle(_StrictModel):
    name: str = Field(min_length=1)
    ic: TestParticleIc


class IntegratorConfig(_StrictModel):
    name: Literal["whfast", "ias15", "mercurius"]
    timestep: Duration | None = None
    divergence_threshold: float | None = Field(default=None, gt=0)
    r_crit_hill: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _timestep_matches_integrator(self) -> IntegratorConfig:
        if self.name in ("whfast", "mercurius") and self.timestep is None:
            raise ValueError(
                f"integrator {self.name!r} is fixed-step and requires 'timestep'"
            )
        if self.name == "ias15" and self.timestep is not None:
            raise ValueError(
                "integrator 'ias15' is adaptive; remove 'timestep'"
            )
        return self


class OutputConfig(_StrictModel):
    format: Literal["parquet"] = "parquet"
    cadence: Duration
    path: str | None = None
    checkpoint: bool = False


class Scenario(_StrictModel):
    schema_version: Literal[1]
    name: str = Field(min_length=1)
    epoch: Epoch
    duration: Duration
    integrator: IntegratorConfig
    bodies: list[Body] = Field(min_length=1)
    test_particles: list[TestParticle] = Field(default_factory=list)
    output: OutputConfig

    @model_validator(mode="after")
    def _unique_names(self) -> Scenario:
        names = [b.name for b in self.bodies] + [p.name for p in self.test_particles]
        seen: set[str] = set()
        dupes: set[str] = set()
        for n in names:
            if n in seen:
                dupes.add(n)
            seen.add(n)
        if dupes:
            raise ValueError(
                f"duplicate names across bodies/test_particles: {sorted(dupes)}"
            )
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> Scenario:
        path = Path(path)
        try:
            with path.open("r", encoding="utf-8") as f:
                raw = yaml.safe_load(f)
        except OSError as e:
            raise ScenarioValidationError(f"could not read {path}: {e}") from e
        if not isinstance(raw, dict):
            raise ScenarioValidationError(
                f"scenario file {path} must contain a YAML mapping at top level"
            )
        raw = _migrate(raw)
        try:
            return cls.model_validate(raw)
        except ValidationError as e:
            raise ScenarioValidationError(
                f"scenario {path} failed validation:\n{e}"
            ) from e


def _migrate(raw: dict[str, Any]) -> dict[str, Any]:
    """Run ordered migrations from the file's declared schema_version to current.

    Only v1 exists today, so this is a gate, not a migration. When schema_version
    bumps, add `migrate_v1_to_v2(raw) -> raw` functions and chain them here.
    """
    v = raw.get("schema_version")
    if v is None:
        raise ScenarioValidationError(
            "scenario is missing required top-level 'schema_version' key"
        )
    if v != SCHEMA_VERSION:
        raise ScenarioValidationError(
            f"scenario schema_version={v!r} not supported "
            f"(current: {SCHEMA_VERSION}); no migrations defined yet"
        )
    return raw
