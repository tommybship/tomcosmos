"""tomcosmos — solar system state simulator.

Public API; anything not re-exported here is off-surface and may change
without notice until 1.0.0 (no earlier than M6).
"""
from tomcosmos.constants import BODY_CONSTANTS, BodyConstant, resolve_body_constant
from tomcosmos.exceptions import (
    DirtyWorkingTreeError,
    EphemerisOutOfRangeError,
    IntegratorDivergedError,
    KernelDriftError,
    ScenarioValidationError,
    TomcosmosError,
    UnknownBodyError,
)
from tomcosmos.io.diagnostics import RunMetadata
from tomcosmos.io.history import StateHistory, load_run
from tomcosmos.runner import run
from tomcosmos.state.scenario import (
    SCHEMA_VERSION,
    Body,
    EphemerisIc,
    ExplicitIc,
    IntegratorConfig,
    OutputConfig,
    Scenario,
    TestParticle,
    TestParticleExplicitIc,
    TestParticleKeplerianIc,
    TestParticleLagrangeIc,
)

__version__ = "0.0.1"

__all__ = [
    "BODY_CONSTANTS",
    "Body",
    "BodyConstant",
    "DirtyWorkingTreeError",
    "EphemerisIc",
    "EphemerisOutOfRangeError",
    "ExplicitIc",
    "IntegratorConfig",
    "IntegratorDivergedError",
    "KernelDriftError",
    "OutputConfig",
    "RunMetadata",
    "SCHEMA_VERSION",
    "Scenario",
    "ScenarioValidationError",
    "StateHistory",
    "TestParticle",
    "TestParticleExplicitIc",
    "TestParticleKeplerianIc",
    "TestParticleLagrangeIc",
    "TomcosmosError",
    "UnknownBodyError",
    "__version__",
    "load_run",
    "resolve_body_constant",
    "run",
]
