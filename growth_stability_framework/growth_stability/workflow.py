from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .parameters import AlloySystem, NumericalSettings
from .stability import StabilityResult, scan_stability
from .undercooling import LambdaSearchResult, iterate_interface_temperature


@dataclass(frozen=True)
class WorkflowResult:
    velocity: float
    interface_temperature: float
    lambda_result: LambdaSearchResult
    stability_result: StabilityResult
    temperature_history: list[float]

    @property
    def lamellar_spacing(self) -> float:
        return self.lambda_result.lamellar_spacing

    @property
    def total_undercooling(self) -> float:
        return self.lambda_result.undercooling.total

    @property
    def stable(self) -> bool:
        return self.stability_result.stable


def lambda_grid(settings: NumericalSettings) -> np.ndarray:
    return np.logspace(
        np.log10(settings.lambda_min),
        np.log10(settings.lambda_max),
        settings.lambda_points,
    )


def q_grid(settings: NumericalSettings) -> np.ndarray:
    return np.logspace(
        np.log10(settings.q_min),
        np.log10(settings.q_max),
        settings.q_points,
    )


def run_velocity_case(
    system: AlloySystem,
    velocity: float,
    settings: NumericalSettings | None = None,
    initial_temperature: float | None = None,
) -> WorkflowResult:
    """Run the full Fig. 4 loop for one pulling/growth velocity.

    The sequence is:
    1. update phase/interfacial response through system.phase_for(V,T);
    2. solve the multicomponent solute field for each candidate lambda;
    3. locate lambda by the minimum-undercooling criterion;
    4. update interface temperature until convergence;
    5. solve E_ji(q) and evaluate the marginal-stability function.
    """

    settings = settings or NumericalSettings()
    t_interface, lambda_result, temperature_history = iterate_interface_temperature(
        system=system,
        velocity=velocity,
        lambda_values=lambda_grid(settings),
        fourier_terms=settings.fourier_terms,
        initial_temperature=initial_temperature,
        tolerance=settings.temperature_tolerance,
        max_iterations=settings.max_temperature_iterations,
    )
    stability_result = scan_stability(
        lambda_result.solute_solution,
        system.physical,
        q_grid(settings),
    )
    return WorkflowResult(
        velocity=float(velocity),
        interface_temperature=float(t_interface),
        lambda_result=lambda_result,
        stability_result=stability_result,
        temperature_history=temperature_history,
    )


def scan_velocity_range(
    system: AlloySystem,
    velocities: np.ndarray,
    settings: NumericalSettings | None = None,
    initial_temperature: float | None = None,
) -> list[WorkflowResult]:
    """Run the complete workflow for several velocities."""

    results: list[WorkflowResult] = []
    next_initial = initial_temperature
    for velocity in velocities:
        result = run_velocity_case(system, float(velocity), settings=settings, initial_temperature=next_initial)
        results.append(result)
        next_initial = result.interface_temperature
    return results


def find_transition_velocity(
    results: list[WorkflowResult],
) -> tuple[float, float] | None:
    """Return the bracketing velocities around the first stable/unstable change."""

    if len(results) < 2:
        return None
    for previous, current in zip(results[:-1], results[1:]):
        if previous.stable != current.stable:
            return previous.velocity, current.velocity
    return None
