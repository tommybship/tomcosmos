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


class DeltaV(_StrictModel):
    """Instantaneous velocity change applied to a body or test particle.

    The integration loop walks the sorted timeline of all Δv events,
    integrates to each `t_offset` (measured from `scenario.epoch`),
    adds `dv` to the target's velocity, then resumes — REBOUND
    handles the discontinuity correctly across all integrators.

    `frame` matches the IC frames; the runtime applies the same
    transform pipeline that resolves explicit ICs, so `(0, 0, 5)`
    in `ecliptic_j2000_barycentric` is rotated by the obliquity
    before being added to the particle's ICRF velocity.
    """

    t_offset: Duration
    dv: tuple[float, float, float]
    frame: Frame = "icrf_barycentric"


class Body(_StrictModel):
    name: str = Field(min_length=1)
    spice_id: int | None = None
    mass_kg: float | None = Field(default=None, gt=0)
    radius_km: float | None = Field(default=None, gt=0)
    ic: BodyIc
    dv_events: list[DeltaV] = Field(default_factory=list)


class TestParticleExplicitIc(_StrictModel):
    type: Literal["explicit"]
    r: tuple[float, float, float]
    v: tuple[float, float, float]
    frame: Frame = "icrf_barycentric"


class TestParticleLagrangeIc(_StrictModel):
    """Place a test particle at one of L1-L5 of a primary/secondary pair.

    L4/L5 have closed-form positions (±60° from the secondary in its
    orbital plane around the primary). L1/L2/L3 are roots of the
    co-linear quintic; we solve numerically. Both primary and
    secondary are looked up from the scenario's bodies at `epoch`,
    so this only works once the integrator has resolved them.
    """

    type: Literal["lagrange"]
    point: Literal["L1", "L2", "L3", "L4", "L5"]
    primary: str = Field(min_length=1)
    secondary: str = Field(min_length=1)


class TestParticleKeplerianIc(_StrictModel):
    """Six-element osculating Keplerian orbit around `parent`.

    Angles are degrees. `epoch_offset` is optional and defaults to 0,
    meaning the elements are valid at `scenario.epoch`. Set it for
    bodies whose elements were measured at a different time (M5
    will use this for SBDB ingest where each asteroid's elements
    have their own epoch); for now the resolver kepler-propagates
    from element-epoch to scenario.epoch when nonzero.
    """

    type: Literal["keplerian"]
    parent: str = Field(min_length=1)
    a_km: float = Field(gt=0)
    e: float = Field(ge=0, lt=1)  # bound orbits only; M5 may relax
    inc_deg: float
    raan_deg: float       # longitude of ascending node Ω
    argp_deg: float       # argument of periapsis ω
    mean_anom_deg: float  # mean anomaly M at the element epoch
    epoch_offset_s: float = 0.0
    frame: Literal[
        "icrf_barycentric",
        "ecliptic_j2000_barycentric",
    ] = "ecliptic_j2000_barycentric"


TestParticleIc = Annotated[
    TestParticleExplicitIc | TestParticleLagrangeIc | TestParticleKeplerianIc,
    Field(discriminator="type"),
]


class TestParticle(_StrictModel):
    name: str = Field(min_length=1)
    ic: TestParticleIc
    dv_events: list[DeltaV] = Field(default_factory=list)


class IntegratorConfig(_StrictModel):
    name: Literal["whfast", "ias15", "mercurius"]
    timestep: Duration | None = None
    divergence_threshold: float | None = Field(default=None, gt=0)
    r_crit_hill: float | None = Field(default=None, gt=0)
    effects: list[Literal["gr"]] = Field(default_factory=list)

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

    @model_validator(mode="after")
    def _effects_unique(self) -> IntegratorConfig:
        if len(self.effects) != len(set(self.effects)):
            raise ValueError(f"duplicate effects in list: {self.effects}")
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

    @model_validator(mode="after")
    def _dv_events_inside_duration(self) -> Scenario:
        """Δv times must lie strictly inside (0, duration). t=0 burns are
        better expressed by adjusting the IC velocity; t≥duration burns
        never fire."""
        duration_s = float(self.duration.to(u.s).value)
        for entity_list, role in (
            (self.bodies, "body"),
            (self.test_particles, "test_particle"),
        ):
            for entity in entity_list:
                for ev in entity.dv_events:
                    t_s = float(ev.t_offset.to(u.s).value)
                    if not (0.0 < t_s < duration_s):
                        raise ValueError(
                            f"{role} {entity.name!r}: dv event t_offset "
                            f"{t_s} s must be in (0, {duration_s} s)"
                        )
        return self

    @model_validator(mode="after")
    def _gr_requires_sun(self) -> Scenario:
        if "gr" in self.integrator.effects:
            body_names = {b.name.lower() for b in self.bodies}
            if "sun" not in body_names:
                raise ValueError(
                    "integrator.effects=['gr'] requires a body named 'sun' "
                    "(GR 1PN correction treats it as the dominant mass)"
                )
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> Scenario:
        path = Path(path)
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as e:
            raise ScenarioValidationError(f"could not read {path}: {e}") from e
        try:
            return cls._from_yaml_string(text)
        except ScenarioValidationError as e:
            raise ScenarioValidationError(f"scenario {path}: {e}") from e

    @classmethod
    def from_yaml_string(cls, text: str) -> Scenario:
        """Parse a scenario from an in-memory YAML string.

        Used by `load_run()` to reconstruct the scenario embedded in a
        Parquet file's metadata without touching disk.
        """
        return cls._from_yaml_string(text)

    @classmethod
    def _from_yaml_string(cls, text: str) -> Scenario:
        raw = yaml.safe_load(text)
        if not isinstance(raw, dict):
            raise ScenarioValidationError(
                "scenario YAML must be a mapping at top level"
            )
        raw = _migrate(raw)
        try:
            return cls.model_validate(raw)
        except ValidationError as e:
            raise ScenarioValidationError(f"scenario failed validation:\n{e}") from e

    def to_yaml_string(self) -> str:
        """Canonical YAML form — embedded in Parquet file metadata and hashed
        for `scenario_sha256`. Keys are sorted so the hash is stable across
        arbitrary insertion orders."""
        return yaml.safe_dump(self.model_dump(mode="json"), sort_keys=True)


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
