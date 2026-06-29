from __future__ import annotations

import numpy as np

from growth_stability import AlloySystem, NumericalSettings, PhaseParameters, PhysicalParameters


def build_system() -> AlloySystem:
    """Return the alloy system to be calculated.

    Fill this function with user-supplied components, nominal composition,
    phase fractions, partition coefficients, diffusion matrix, liquidus slopes,
    Gibbs-Thomson coefficients, thermal properties, and optional kinetic data.
    """

    # TODO 1: replace with the component names of your alloy system.
    # components = ("Component_1", "Component_2", "Component_3")

    # TODO 2: replace with nominal mole fractions or atomic fractions.
    # The values must have the same length as components and must sum to 1.
    # nominal_composition = np.array([...], dtype=float)

    # TODO 3: replace with phase fractions and partition coefficients.
    # k_alpha and k_beta must have the same length as components.
    # phase = PhaseParameters(
    #     f_alpha=...,
    #     f_beta=...,
    #     k_alpha=np.array([...], dtype=float),
    #     k_beta=np.array([...], dtype=float),
    # )

    # TODO 4: replace with thermodynamic, diffusion, capillary, and thermal inputs.
    # diffusion_matrix must be an N x N matrix for an N-component alloy.
    # liquidus_slopes, liquidus_slopes_alpha, liquidus_slopes_beta, and
    # diffusion_speed_liquid must have the same length as components.
    # physical = PhysicalParameters(
    #     temperature_eutectic=...,
    #     diffusion_matrix=np.array([...], dtype=float),
    #     diffusion_speed_liquid=np.array([...], dtype=float),
    #     liquidus_slopes=np.array([...], dtype=float),
    #     liquidus_slopes_alpha=np.array([...], dtype=float),
    #     liquidus_slopes_beta=np.array([...], dtype=float),
    #     gamma=...,
    #     gamma_alpha=...,
    #     gamma_beta=...,
    #     theta_alpha=...,
    #     theta_beta=...,
    #     thermal_conductivity_solid=...,
    #     thermal_conductivity_liquid=...,
    #     thermal_gradient_solid=...,
    #     thermal_gradient_liquid=...,
    # )

    # TODO 5: after filling TODO 1-4, remove the NotImplementedError below and return:
    # return AlloySystem(
    #     components=components,
    #     nominal_composition=nominal_composition,
    #     phase=phase,
    #     physical=physical,
    # )

    raise NotImplementedError(
        "Please provide alloy-specific parameters in examples/system_template.py "
        "before running the workflow."
    )


def velocity_grid() -> np.ndarray:
    """Return the growth/pulling velocities for the calculation."""

    # TODO 6: replace with the velocities to be scanned, in m/s.
    # return np.array([...], dtype=float)

    raise NotImplementedError(
        "Please provide the velocity grid for your own system in examples/system_template.py."
    )


def numerical_settings() -> NumericalSettings:
    """Numerical controls for the full workflow.

    These values control resolution only; they are not alloy-specific.
    """

    return NumericalSettings(
        fourier_terms=1000,
        lambda_min=2.0e-7,
        lambda_max=2.0e-5,
        lambda_points=160,
        q_min=1.0e4,
        q_max=1.0e10,
        q_points=240,
        max_temperature_iterations=80,
        temperature_tolerance=1.0e-4,
    )
