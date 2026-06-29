from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .parameters import AlloySystem, GrowthConditions, PhaseParameters
from .solute_field import SoluteFieldSolution, solve_solute_field


@dataclass(frozen=True)
class UndercoolingBreakdown:
    constitutional_dendritic: float
    constitutional_eutectic: float
    curvature: float
    kinetic: float

    @property
    def total(self) -> float:
        return self.constitutional_dendritic + self.constitutional_eutectic + self.curvature + self.kinetic


@dataclass(frozen=True)
class LambdaSearchResult:
    lamellar_spacing: float
    undercooling: UndercoolingBreakdown
    solute_solution: SoluteFieldSolution
    searched_lambdas: np.ndarray
    searched_total_undercooling: np.ndarray


def phase_interval_averages(
    solute: SoluteFieldSolution,
    n_points: int = 2001,
) -> tuple[np.ndarray, np.ndarray]:
    """Average interface compositions over alpha and beta lamella intervals."""

    x = np.linspace(0.0, solute.lamellar_spacing, n_points)
    c = solute.concentration(x, np.zeros_like(x))
    alpha_end = solute.phase.f_alpha * solute.lamellar_spacing
    alpha_mask = x <= alpha_end
    beta_mask = x > alpha_end
    if not np.any(beta_mask):
        raise ValueError("No beta interval points; increase n_points or check f_alpha.")
    c_alpha = np.mean(c[:, alpha_mask], axis=1)
    c_beta = np.mean(c[:, beta_mask], axis=1)
    return c_alpha, c_beta


def harmonic_liquidus_slope(m_alpha: np.ndarray, m_beta: np.ndarray) -> np.ndarray:
    denom = m_alpha + m_beta
    safe = np.where(np.abs(denom) < 1e-30, np.inf, denom)
    return m_alpha * m_beta / safe


def capillarity_constants(system: AlloySystem, phase: PhaseParameters) -> np.ndarray:
    """Lamellar capillarity constants used in Eq. (25).

    The expression follows the manuscript role of gamma_alpha, gamma_beta,
    theta_alpha, theta_beta, phase fractions, and phase-specific slopes. The
    sign of the liquidus slopes is retained through the slopes supplied by the
    user.
    """

    p = system.physical
    m_alpha = p.liquidus_slopes_alpha
    m_beta = p.liquidus_slopes_beta
    eps = 1e-30
    alpha_term = np.divide(
        p.gamma_alpha * np.sin(p.theta_alpha),
        np.abs(phase.f_alpha * m_alpha),
        out=np.zeros_like(m_alpha, dtype=float),
        where=np.abs(m_alpha) > eps,
    )
    beta_term = np.divide(
        p.gamma_beta * np.sin(p.theta_beta),
        np.abs(phase.f_beta * m_beta),
        out=np.zeros_like(m_beta, dtype=float),
        where=np.abs(m_beta) > eps,
    )
    return 2.0 * (alpha_term + beta_term)


def undercooling_for_solution(
    system: AlloySystem,
    solute: SoluteFieldSolution,
    tip_radius: float | None = None,
) -> UndercoolingBreakdown:
    """Evaluate Eqs. (19)-(27) for an existing solute-field solution."""

    p = system.physical
    phase = solute.phase
    c_interface_liquid = solute.interface_average
    c_alpha, c_beta = phase_interval_averages(solute)

    # Eq. (20): long-range constitutional undercooling.
    constitutional_dendritic = float(np.sum(p.liquidus_slopes * (system.nominal_composition - c_interface_liquid)))

    # Eq. (24): short-range eutectic constitutional undercooling.
    m_alpha = p.liquidus_slopes_alpha
    m_beta = p.liquidus_slopes_beta
    denom = m_alpha + m_beta
    safe = np.where(np.abs(denom) < 1e-30, np.inf, denom)
    constitutional_eutectic = float(np.sum(m_alpha * m_beta * (c_alpha + c_beta - 2.0 * system.nominal_composition) / safe))

    # Eq. (25): dendrite-tip curvature plus local lamellar curvature.
    cap_const = capillarity_constants(system, phase)
    m_harmonic = harmonic_liquidus_slope(m_alpha, m_beta)
    curvature_lamellar = float(np.sum(np.abs(m_harmonic) * cap_const / solute.lamellar_spacing))
    curvature_tip = 0.0 if tip_radius is None or np.isinf(tip_radius) else 2.0 * p.gamma / tip_radius

    # Eqs. (26)-(27): kinetic undercooling.
    kinetic_coeff = np.asarray(p.kinetic_coefficients, dtype=float)
    kinetic = float(np.sum(np.where(np.isfinite(kinetic_coeff), solute.velocity / kinetic_coeff, 0.0)))

    return UndercoolingBreakdown(
        constitutional_dendritic=constitutional_dendritic,
        constitutional_eutectic=constitutional_eutectic,
        curvature=curvature_tip + curvature_lamellar,
        kinetic=kinetic,
    )


def find_minimum_undercooling_spacing(
    system: AlloySystem,
    velocity: float,
    interface_temperature: float,
    lambda_values: np.ndarray,
    fourier_terms: int,
) -> LambdaSearchResult:
    """Find lamellar spacing using the minimum-undercooling criterion."""

    totals = np.zeros_like(lambda_values, dtype=float)
    solutions: list[SoluteFieldSolution] = []
    breakdowns: list[UndercoolingBreakdown] = []
    phase = system.phase_for(velocity, interface_temperature)
    for idx, lam in enumerate(lambda_values):
        conditions = GrowthConditions(
            velocity=velocity,
            lamellar_spacing=float(lam),
            temperature_gradient=system.physical.thermal_gradient_liquid,
            interface_temperature=interface_temperature,
        )
        sol = solve_solute_field(system, conditions, fourier_terms=fourier_terms, phase=phase)
        breakdown = undercooling_for_solution(system, sol)
        solutions.append(sol)
        breakdowns.append(breakdown)
        totals[idx] = breakdown.total

    best = int(np.nanargmin(totals))
    return LambdaSearchResult(
        lamellar_spacing=float(lambda_values[best]),
        undercooling=breakdowns[best],
        solute_solution=solutions[best],
        searched_lambdas=lambda_values,
        searched_total_undercooling=totals,
    )


def iterate_interface_temperature(
    system: AlloySystem,
    velocity: float,
    lambda_values: np.ndarray,
    fourier_terms: int,
    initial_temperature: float | None = None,
    tolerance: float = 1e-4,
    max_iterations: int = 80,
) -> tuple[float, LambdaSearchResult, list[float]]:
    """Implement the Fig. 4 interface-temperature loop."""

    t_old = float(initial_temperature or system.physical.temperature_eutectic)
    history: list[float] = [t_old]
    last_result: LambdaSearchResult | None = None
    for _ in range(max_iterations):
        result = find_minimum_undercooling_spacing(
            system,
            velocity=velocity,
            interface_temperature=t_old,
            lambda_values=lambda_values,
            fourier_terms=fourier_terms,
        )
        t_new = system.physical.temperature_eutectic - result.undercooling.total
        history.append(float(t_new))
        last_result = result
        if abs(t_new - t_old) < tolerance:
            return float(t_new), result, history
        t_old = float(t_new)

    if last_result is None:
        raise RuntimeError("Interface-temperature iteration did not run.")
    return float(history[-1]), last_result, history
