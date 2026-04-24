class TomcosmosError(Exception):
    """Base exception for tomcosmos."""


class ScenarioValidationError(TomcosmosError):
    """Raised when a scenario YAML fails schema or semantic validation."""


class UnknownBodyError(TomcosmosError):
    """Raised when a body name or SPICE ID isn't in `constants.BODY_CONSTANTS`."""


class IntegratorDivergedError(TomcosmosError):
    """Raised when the integrator's energy error exceeds the configured threshold."""


class DirtyWorkingTreeError(TomcosmosError):
    """Raised when a run is attempted against a dirty git working tree without --allow-dirty."""


class EphemerisOutOfRangeError(TomcosmosError):
    """Raised when a scenario requests epochs outside the loaded ephemeris coverage."""


class KernelDriftError(TomcosmosError):
    """Raised when a kernel's SHA256 disagrees with the committed manifest."""
