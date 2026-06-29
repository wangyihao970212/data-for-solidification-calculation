from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .parameters import PhysicalParameters
from .solute_field import SoluteFieldSolution


@dataclass(frozen=True)
class StabilityResult:
    q_values: np.ndarray
    growth_rates: np.ndarray
    e_matrices: np.ndarray
    max_growth_rate: float
    q_at_max: float
    stable: bool


def _thermal_denominator(physical: PhysicalParameters) -> float:
    return (
        physical.thermal_conductivity_solid * physical.thermal_gradient_solid
        - physical.thermal_conductivity_liquid * physical.thermal_gradient_liquid
    )


def _thermal_average(physical: PhysicalParameters) -> float:
    return (
        physical.thermal_conductivity_solid * physical.thermal_gradient_solid
        + physical.thermal_conductivity_liquid * physical.thermal_gradient_liquid
    ) / (physical.thermal_conductivity_solid + physical.thermal_conductivity_liquid)


def solve_e_matrix(
    solute: SoluteFieldSolution,
    physical: PhysicalParameters,
    q: float,
    delta_c_star: np.ndarray | None = None,
) -> np.ndarray:
    """Solve E_ji for one perturbation wavenumber.

    This implements the Hunziker plane-front stability construction used in
    the manuscript: E_ji=N_ji|E_i|, where the signed magnitudes |E_i| are
    obtained from the compatibility of temperature and solute fields.

    H_ki in Hunziker's notation is supplied as physical.tie_line_derivative.
    When no tie-line sensitivity is available, H=0 is used, matching the
    common first implementation and the model assumptions in the manuscript.
    """

    n = len(solute.components)
    d = physical.diffusion_matrix
    h = physical.tie_line_derivative
    m_slopes = physical.liquidus_slopes
    b_vals = solute.eigenvalues
    n_vec = solute.eigenvectors
    a = solute.a_matrix
    velocity = solute.velocity

    if delta_c_star is None:
        delta_c_star = solute.interface_average - solute.nominal_composition
    delta_c_star = np.asarray(delta_c_star, dtype=float)
    if delta_c_star.shape[0] != n:
        raise ValueError("delta_c_star length must match component count.")

    denom = _thermal_denominator(physical)
    if abs(denom) < 1e-16:
        raise ValueError(
            "Hunziker E_ji system is singular because kappa_s*G_s ~= kappa_l*G_l. "
            "Use physically distinct solid/liquid thermal parameters or provide a regularized model."
        )

    kappa_sum = physical.thermal_conductivity_solid + physical.thermal_conductivity_liquid
    thermal_avg = _thermal_average(physical)
    sqrt_terms = np.sqrt(1.0 + (2.0 * q * b_vals / velocity) ** 2)
    a_over_b = a / b_vals.reshape((1, -1))
    slope_a_over_b = float(np.sum(m_slopes[:, None] * a_over_b))

    coeff = np.zeros((n, n), dtype=float)
    rhs = np.zeros(n, dtype=float)

    for k in range(n):
        for mode in range(n):
            total = 0.0
            for i in range(n):
                mass_term = d[k, i] / (2.0 * b_vals[mode]) * (1.0 + sqrt_terms[mode])
                tie_term = h[k, i]
                thermal_coupling = m_slopes[i] * delta_c_star[k] * q * kappa_sum / denom
                total += n_vec[i, mode] * (mass_term - tie_term - thermal_coupling)
            coeff[k, mode] = total

        rhs_thermal = -delta_c_star[k] * q * kappa_sum / denom * (
            physical.gamma * q**2 + thermal_avg + velocity * slope_a_over_b
        )
        rhs_flux = 0.0
        for i in range(n):
            for mode in range(n):
                rhs_flux += velocity * a[i, mode] / b_vals[mode] * (d[k, i] / b_vals[mode] - h[k, i])
        rhs[k] = rhs_thermal + rhs_flux

    try:
        magnitudes = np.linalg.solve(coeff, rhs)
    except np.linalg.LinAlgError:
        magnitudes = np.linalg.lstsq(coeff, rhs, rcond=None)[0]
    return n_vec @ np.diag(magnitudes)


def stability_function(
    solute: SoluteFieldSolution,
    physical: PhysicalParameters,
    q: float,
    e_matrix: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    """Return the Eq. (17) perturbation growth rate for one q."""

    e = e_matrix if e_matrix is not None else solve_e_matrix(solute, physical, q)
    b_vals = solute.eigenvalues
    a = solute.a_matrix
    diffusion_term = float(np.sum(physical.liquidus_slopes[:, None] * (e - solute.velocity * a / b_vals)))
    growth_rate = diffusion_term - physical.gamma * q**2 - _thermal_average(physical)
    return growth_rate, e


def scan_stability(
    solute: SoluteFieldSolution,
    physical: PhysicalParameters,
    q_values: np.ndarray,
) -> StabilityResult:
    """Scan perturbation wavenumbers and report whether all modes are stable."""

    q_values = np.asarray(q_values, dtype=float)
    growth_rates = np.full_like(q_values, np.nan, dtype=float)
    e_matrices = np.full((len(q_values), len(solute.components), len(solute.components)), np.nan, dtype=float)
    for idx, q in enumerate(q_values):
        try:
            growth_rates[idx], e_matrices[idx] = stability_function(solute, physical, q)
        except (FloatingPointError, ValueError, np.linalg.LinAlgError):
            continue

    if np.all(np.isnan(growth_rates)):
        raise RuntimeError("No valid perturbation mode was evaluated in the stability scan.")
    max_idx = int(np.nanargmax(growth_rates))
    max_growth_rate = float(growth_rates[max_idx])
    return StabilityResult(
        q_values=q_values,
        growth_rates=growth_rates,
        e_matrices=e_matrices,
        max_growth_rate=max_growth_rate,
        q_at_max=float(q_values[max_idx]),
        stable=max_growth_rate < 0.0,
    )
