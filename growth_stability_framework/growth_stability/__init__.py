"""Growth-stability framework for complex-component eutectics."""

from .parameters import (
    AlloySystem,
    GrowthConditions,
    InterfaceKinetics,
    NumericalSettings,
    PhysicalParameters,
    PhaseParameters,
)
from .solute_field import SoluteFieldSolution, solve_solute_field
from .stability import StabilityResult, scan_stability, solve_e_matrix
from .workflow import WorkflowResult, find_transition_velocity, run_velocity_case, scan_velocity_range

__all__ = [
    "AlloySystem",
    "GrowthConditions",
    "InterfaceKinetics",
    "NumericalSettings",
    "PhysicalParameters",
    "PhaseParameters",
    "SoluteFieldSolution",
    "StabilityResult",
    "WorkflowResult",
    "find_transition_velocity",
    "run_velocity_case",
    "scan_stability",
    "scan_velocity_range",
    "solve_e_matrix",
    "solve_solute_field",
]
