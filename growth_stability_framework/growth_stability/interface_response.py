from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


R_GAS = 8.31446261815324


@dataclass(frozen=True)
class PartialDragResult:
    liquid_composition: np.ndarray
    solid_composition: np.ndarray
    effective_composition: np.ndarray
    residual: np.ndarray
    converged: bool
    iterations: int
    residual_norm: float


def normalize_composition(composition: np.ndarray, minimum: float = 1e-12) -> np.ndarray:
    comp = np.asarray(composition, dtype=float)
    clipped = np.maximum(comp, minimum)
    return clipped / np.sum(clipped)


def chemical_potentials_from_gibbs(
    gibbs_energy: Callable[[np.ndarray, float], float],
    composition: np.ndarray,
    temperature: float,
    step: float = 1e-6,
) -> np.ndarray:
    """Return diffusion chemical potentials from a molar Gibbs-energy function.

    For a phase molar Gibbs energy G(C,T), the component chemical potential is
    evaluated as mu_i = G + dG/dC_i - sum_j C_j dG/dC_j. This matches the
    finite-difference construction used in the earlier draft scripts.
    """

    composition = normalize_composition(composition)
    n = composition.size
    derivatives = np.zeros(n, dtype=float)
    for i in range(n):
        plus = composition.copy()
        minus = composition.copy()
        plus[i] += step
        minus[i] -= step
        plus = normalize_composition(plus)
        minus = normalize_composition(minus)
        derivatives[i] = (gibbs_energy(plus, temperature) - gibbs_energy(minus, temperature)) / (2.0 * step)
    g = gibbs_energy(composition, temperature)
    return g + derivatives - float(np.dot(composition, derivatives))


def effective_interface_composition(
    liquid_composition: np.ndarray,
    solid_composition: np.ndarray,
    lambda_sd: float,
) -> np.ndarray:
    """C_eff = lambda_sd C_L + (1-lambda_sd) C_S in the partial-drag model."""

    liquid = np.asarray(liquid_composition, dtype=float)
    solid = np.asarray(solid_composition, dtype=float)
    return lambda_sd * liquid + (1.0 - lambda_sd) * solid


def partial_drag_residual(
    liquid_composition: np.ndarray,
    solid_composition: np.ndarray,
    temperature: float,
    velocity: float,
    diffusion_speed: float,
    interface_diffusion_speed: float,
    lambda_sd: float,
    chemical_potential_liquid: Callable[[np.ndarray, float], np.ndarray],
    chemical_potential_solid: Callable[[np.ndarray, float], np.ndarray],
    solute_indices: tuple[int, ...] | None = None,
) -> np.ndarray:
    """Residual of the energy-dissipation-based partial-drag equations.

    The first equation is the interface response relation:
        V = V0/(RT) * sum_i C_eff,i * (mu_i^L - mu_i^S)
    The remaining equations are component flux/drag balances for independent
    solutes. By default the solvent is component 0 and the independent solutes
    are components 1...N-1.
    """

    liquid = normalize_composition(liquid_composition)
    solid = normalize_composition(solid_composition)
    c_eff = effective_interface_composition(liquid, solid, lambda_sd)
    delta_mu = chemical_potential_liquid(liquid, temperature) - chemical_potential_solid(solid, temperature)

    if solute_indices is None:
        solute_indices = tuple(range(1, liquid.size))

    residuals = [
        velocity - diffusion_speed / (R_GAS * temperature) * float(np.dot(c_eff, delta_mu))
    ]
    for idx in solute_indices:
        residuals.append(
            velocity * (c_eff[idx] - solid[idx])
            - c_eff[idx] * interface_diffusion_speed * (
                velocity / diffusion_speed - delta_mu[idx] / (R_GAS * temperature)
            )
        )
    return np.asarray(residuals, dtype=float)


def _default_variable_layout(
    n_components: int,
    fixed_liquid: dict[int, float] | None = None,
    fixed_solid: dict[int, float] | None = None,
) -> list[tuple[str, int]]:
    fixed_liquid = fixed_liquid or {}
    fixed_solid = fixed_solid or {}
    layout: list[tuple[str, int]] = []
    for idx in range(1, n_components):
        if idx not in fixed_liquid:
            layout.append(("L", idx))
    for idx in range(1, n_components):
        if idx not in fixed_solid:
            layout.append(("S", idx))
    return layout


def _variables_to_compositions(
    values: np.ndarray,
    n_components: int,
    variable_layout: list[tuple[str, int]],
    fixed_liquid: dict[int, float] | None = None,
    fixed_solid: dict[int, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Map independent solute variables to full liquid/solid compositions."""

    fixed_liquid = fixed_liquid or {}
    fixed_solid = fixed_solid or {}
    liquid = np.zeros(n_components, dtype=float)
    solid = np.zeros(n_components, dtype=float)
    for idx, value in fixed_liquid.items():
        liquid[idx] = value
    for idx, value in fixed_solid.items():
        solid[idx] = value
    for value, (phase_name, idx) in zip(values, variable_layout):
        if phase_name == "L":
            liquid[idx] = value
        else:
            solid[idx] = value
    liquid[0] = 1.0 - float(np.sum(liquid[1:]))
    solid[0] = 1.0 - float(np.sum(solid[1:]))
    return normalize_composition(liquid), normalize_composition(solid)


def _initial_variables(
    liquid: np.ndarray,
    solid: np.ndarray,
    variable_layout: list[tuple[str, int]],
) -> np.ndarray:
    values = []
    for phase_name, idx in variable_layout:
        values.append(liquid[idx] if phase_name == "L" else solid[idx])
    return np.asarray(values, dtype=float)


def solve_partial_drag_response(
    initial_liquid_composition: np.ndarray,
    initial_solid_composition: np.ndarray,
    temperature: float,
    velocity: float,
    diffusion_speed: float,
    interface_diffusion_speed: float,
    lambda_sd: float,
    chemical_potential_liquid: Callable[[np.ndarray, float], np.ndarray],
    chemical_potential_solid: Callable[[np.ndarray, float], np.ndarray],
    fixed_liquid_solutes: dict[int, float] | None = None,
    fixed_solid_solutes: dict[int, float] | None = None,
    tolerance: float = 1e-9,
    max_iterations: int = 80,
) -> PartialDragResult:
    """Solve the partial-drag equations with a finite-difference Newton method.

    This avoids requiring SciPy, so the repository remains directly runnable
    with the bundled Python environment. For publication calculations, supply
    the same Gibbs functions or CALPHAD-tabulated chemical-potential callables
    used in the manuscript.
    """

    fixed_liquid_solutes = fixed_liquid_solutes or {}
    fixed_solid_solutes = fixed_solid_solutes or {}
    liquid0 = normalize_composition(initial_liquid_composition)
    solid0 = normalize_composition(initial_solid_composition)
    n = liquid0.size
    variable_layout = _default_variable_layout(n, fixed_liquid_solutes, fixed_solid_solutes)
    values = _initial_variables(liquid0, solid0, variable_layout)

    def residual_from_values(x: np.ndarray) -> np.ndarray:
        liquid, solid = _variables_to_compositions(
            x,
            n,
            variable_layout,
            fixed_liquid=fixed_liquid_solutes,
            fixed_solid=fixed_solid_solutes,
        )
        return partial_drag_residual(
            liquid,
            solid,
            temperature,
            velocity,
            diffusion_speed,
            interface_diffusion_speed,
            lambda_sd,
            chemical_potential_liquid,
            chemical_potential_solid,
        )

    converged = False
    residual = residual_from_values(values)
    for iteration in range(1, max_iterations + 1):
        norm = float(np.linalg.norm(residual))
        if norm < tolerance:
            converged = True
            break

        jacobian = np.zeros((residual.size, values.size), dtype=float)
        for idx in range(values.size):
            step = 1e-6 * max(1.0, abs(values[idx]))
            plus = values.copy()
            minus = values.copy()
            plus[idx] += step
            minus[idx] -= step
            jacobian[:, idx] = (residual_from_values(plus) - residual_from_values(minus)) / (2.0 * step)

        delta = np.linalg.lstsq(jacobian, -residual, rcond=None)[0]
        damping = 1.0
        best_values = values.copy()
        best_residual = residual
        best_norm = norm
        for _ in range(12):
            trial = values + damping * delta
            liquid_trial, solid_trial = _variables_to_compositions(
                trial,
                n,
                variable_layout,
                fixed_liquid=fixed_liquid_solutes,
                fixed_solid=fixed_solid_solutes,
            )
            if np.all(liquid_trial > 0.0) and np.all(solid_trial > 0.0):
                trial_residual = residual_from_values(trial)
                trial_norm = float(np.linalg.norm(trial_residual))
                if trial_norm < best_norm:
                    best_values = trial
                    best_residual = trial_residual
                    best_norm = trial_norm
                    break
            damping *= 0.5
        values = best_values
        residual = best_residual
    else:
        iteration = max_iterations

    liquid, solid = _variables_to_compositions(
        values,
        n,
        variable_layout,
        fixed_liquid=fixed_liquid_solutes,
        fixed_solid=fixed_solid_solutes,
    )
    c_eff = effective_interface_composition(liquid, solid, lambda_sd)
    residual_norm = float(np.linalg.norm(residual))
    return PartialDragResult(
        liquid_composition=liquid,
        solid_composition=solid,
        effective_composition=c_eff,
        residual=residual,
        converged=converged or residual_norm < tolerance,
        iterations=iteration,
        residual_norm=residual_norm,
    )
