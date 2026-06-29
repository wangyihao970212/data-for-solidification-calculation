from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

import numpy as np


ArrayLike = Iterable[float] | np.ndarray


def as_vector(values: ArrayLike, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array.")
    return arr


def as_matrix(values: ArrayLike, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"{name} must be a square two-dimensional array.")
    return arr


@dataclass(frozen=True)
class PhaseParameters:
    """Phase fractions and kinetic partition coefficients.

    k_alpha and k_beta are velocity- and temperature-dependent partition
    coefficients in the manuscript. They can be supplied directly or updated
    by the interfacial response model before each solute-field calculation.
    """

    f_alpha: float
    f_beta: float
    k_alpha: ArrayLike
    k_beta: ArrayLike

    def __post_init__(self) -> None:
        if self.f_alpha <= 0 or self.f_beta <= 0:
            raise ValueError("Phase fractions must be positive.")
        if abs(self.f_alpha + self.f_beta - 1.0) > 1e-6:
            raise ValueError("f_alpha + f_beta must be 1.")
        object.__setattr__(self, "k_alpha", as_vector(self.k_alpha, "k_alpha"))
        object.__setattr__(self, "k_beta", as_vector(self.k_beta, "k_beta"))


@dataclass(frozen=True)
class PhysicalParameters:
    """Physical constants used in undercooling and stability equations."""

    temperature_eutectic: float
    diffusion_matrix: ArrayLike
    diffusion_speed_liquid: ArrayLike
    liquidus_slopes: ArrayLike
    liquidus_slopes_alpha: ArrayLike
    liquidus_slopes_beta: ArrayLike
    gamma: float
    gamma_alpha: float
    gamma_beta: float
    theta_alpha: float
    theta_beta: float
    thermal_conductivity_solid: float
    thermal_conductivity_liquid: float
    thermal_gradient_solid: float
    thermal_gradient_liquid: float
    kinetic_coefficients: ArrayLike | None = None
    tie_line_derivative: ArrayLike | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "diffusion_matrix", as_matrix(self.diffusion_matrix, "diffusion_matrix"))
        n = self.diffusion_matrix.shape[0]
        for name in (
            "diffusion_speed_liquid",
            "liquidus_slopes",
            "liquidus_slopes_alpha",
            "liquidus_slopes_beta",
        ):
            value = as_vector(getattr(self, name), name)
            if value.shape[0] != n:
                raise ValueError(f"{name} length must match diffusion_matrix size.")
            object.__setattr__(self, name, value)
        if self.kinetic_coefficients is None:
            object.__setattr__(self, "kinetic_coefficients", np.full(n, np.inf))
        else:
            object.__setattr__(
                self,
                "kinetic_coefficients",
                as_vector(self.kinetic_coefficients, "kinetic_coefficients"),
            )
        if self.tie_line_derivative is None:
            object.__setattr__(self, "tie_line_derivative", np.zeros((n, n)))
        else:
            h = as_matrix(self.tie_line_derivative, "tie_line_derivative")
            if h.shape != (n, n):
                raise ValueError("tie_line_derivative shape must match diffusion_matrix.")
            object.__setattr__(self, "tie_line_derivative", h)


@dataclass(frozen=True)
class GrowthConditions:
    """Solidification conditions for one calculation point."""

    velocity: float
    lamellar_spacing: float
    temperature_gradient: float = 6000.0
    interface_temperature: float | None = None


@dataclass(frozen=True)
class NumericalSettings:
    """Numerical controls for Fourier truncation and iterative workflow."""

    fourier_terms: int = 1000
    lambda_min: float = 1e-8
    lambda_max: float = 2e-5
    lambda_points: int = 160
    q_min: float = 1e2
    q_max: float = 1e10
    q_points: int = 240
    max_temperature_iterations: int = 80
    temperature_tolerance: float = 1e-4


@dataclass(frozen=True)
class InterfaceKinetics:
    """Partial-drag interfacial response settings.

    This class allows the workflow to use either externally supplied kinetic
    partition coefficients or a callable that returns updated phase parameters
    for each velocity/temperature pair.
    """

    lambda_sd: float = 0.3
    update_phase_parameters: Callable[[float, float, PhaseParameters], PhaseParameters] | None = None


@dataclass(frozen=True)
class AlloySystem:
    """All alloy-specific inputs for the framework."""

    components: tuple[str, ...]
    nominal_composition: ArrayLike
    phase: PhaseParameters
    physical: PhysicalParameters
    kinetics: InterfaceKinetics = field(default_factory=InterfaceKinetics)

    def __post_init__(self) -> None:
        nominal = as_vector(self.nominal_composition, "nominal_composition")
        if abs(float(np.sum(nominal)) - 1.0) > 1e-5:
            raise ValueError("nominal_composition must sum to 1.")
        n = len(self.components)
        if nominal.shape[0] != n:
            raise ValueError("components and nominal_composition lengths differ.")
        if self.phase.k_alpha.shape[0] != n or self.phase.k_beta.shape[0] != n:
            raise ValueError("phase coefficient lengths must match components.")
        if self.physical.diffusion_matrix.shape != (n, n):
            raise ValueError("diffusion_matrix shape must match components.")
        object.__setattr__(self, "nominal_composition", nominal)

    @property
    def n_components(self) -> int:
        return len(self.components)

    def phase_for(self, velocity: float, temperature: float) -> PhaseParameters:
        if self.kinetics.update_phase_parameters is None:
            return self.phase
        return self.kinetics.update_phase_parameters(velocity, temperature, self.phase)
