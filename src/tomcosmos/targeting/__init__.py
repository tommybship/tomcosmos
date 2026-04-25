"""Targeting primitives for mission-planning composition.

This sub-package provides the orbital-mechanics building blocks needed
to compute Δvs that take a spacecraft from a known state to a desired
future state. tomcosmos's philosophy is to ship the primitives, not
the optimization stack on top — a user can call `lambert(...)` to get
the velocity vectors that connect (r₁, t₁) to (r₂, t₂) in a Keplerian
two-body sense, and then construct `DeltaV` events for the scenario
YAML by hand. Iterating that for fuel-optimal multi-target tours,
B-plane targeting, or low-thrust optimization is a different product
(see PLAN.md > Non-goals).
"""
from tomcosmos.targeting.lambert import lambert
from tomcosmos.targeting.transfer import (
    MU_SUN_KM3_S2,
    Transfer,
    compute_transfer,
)

__all__ = [
    "MU_SUN_KM3_S2",
    "Transfer",
    "compute_transfer",
    "lambert",
]
