class TomcosmosError(Exception):
    """Base exception for tomcosmos."""


class ScenarioValidationError(TomcosmosError):
    """Raised when a scenario YAML fails schema or semantic validation."""


class IntegratorDivergedError(TomcosmosError):
    """Raised when the integrator's energy error exceeds the configured threshold."""


class DirtyWorkingTreeError(TomcosmosError):
    """Raised when a run is attempted against a dirty git working tree without --allow-dirty."""


class EphemerisOutOfRangeError(TomcosmosError):
    """Raised when a scenario requests epochs outside the loaded ephemeris coverage."""
