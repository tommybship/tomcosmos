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
from tomcosmos.io.history import StateHistory
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
    "SCHEMA_VERSION",
    "Scenario",
    "ScenarioValidationError",
    "StateHistory",
    "TestParticle",
    "TestParticleExplicitIc",
    "TomcosmosError",
    "UnknownBodyError",
    "__version__",
    "resolve_body_constant",
    "run",
]
