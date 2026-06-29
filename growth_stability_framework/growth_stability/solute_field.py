from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .parameters import AlloySystem, GrowthConditions, PhaseParameters


@dataclass(frozen=True)
class SoluteFieldSolution:
    """Analytical solute-field solution corresponding to Eqs. (1)-(10)."""

    components: tuple[str, ...]
    velocity: float
    lamellar_spacing: float
    phi: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    a_matrix: np.ndarray
    q_tensor: np.ndarray
    omega_modes: np.ndarray
    b_modes: np.ndarray
    nominal_composition: np.ndarray
    phase: PhaseParameters

    def concentration(self, x: np.ndarray | float, z: np.ndarray | float) -> np.ndarray:
        """Return C_j(x,z) for all components.

        The returned array has shape (n_components, *broadcast_shape).
        x and z are in meters.
        """

        x_arr, z_arr = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(z, dtype=float))
        out = np.zeros((len(self.components),) + x_arr.shape, dtype=float)
        for j in range(len(self.components)):
            out[j, ...] = self.nominal_composition[j]

        for i, b_i in enumerate(self.eigenvalues):
            out += self.a_matrix[:, i].reshape((-1,) + (1,) * x_arr.ndim) * np.exp(
                -self.velocity * z_arr / b_i
            )

        for m_idx, b_m in enumerate(self.b_modes):
            cos_term = np.cos(b_m * x_arr)
            for i, b_i in enumerate(self.eigenvalues):
                exp_term = np.exp(-self.omega_modes[m_idx, i] * self.velocity * z_arr / b_i)
                out += self.q_tensor[m_idx, :, i].reshape((-1,) + (1,) * x_arr.ndim) * cos_term * exp_term
        return out

    def interface_profile(self, n_points: int = 501, periods: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
        x = np.linspace(0.0, periods * self.lamellar_spacing, n_points)
        z = np.zeros_like(x)
        return x, self.concentration(x, z)

    def z_profile(self, z_max: float, n_points: int = 501, x: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        z = np.linspace(0.0, z_max, n_points)
        x_arr = np.full_like(z, x)
        return z, self.concentration(x_arr, z)

    @property
    def interface_average(self) -> np.ndarray:
        """Average liquid composition normal to the interface, C_inf + sum_i A_ji."""

        return self.nominal_composition + np.sum(self.a_matrix, axis=1)


def nonequilibrium_phi(velocity: float, diffusion_speed_liquid: np.ndarray) -> np.ndarray:
    """Compute phi_i = 1 - (V / V_i^L)^2 from Eq. (1)."""

    phi = 1.0 - (velocity / diffusion_speed_liquid) ** 2
    return np.maximum(phi, 1e-12)


def eigensystem(d_phi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eigenvalues/eigenvectors of D*phi, sorted by decreasing eigenvalue."""

    values, vectors = np.linalg.eig(d_phi)
    if np.max(np.abs(values.imag)) > 1e-8 * max(1.0, np.max(np.abs(values.real))):
        raise ValueError("Diffusion matrix produced complex eigenvalues with non-negligible imaginary parts.")
    values = values.real
    vectors = vectors.real
    if np.any(values <= 0):
        raise ValueError("All eigenvalues of D*phi must be positive for the analytical solution.")

    order = np.argsort(values)[::-1]
    values = values[order]
    vectors = vectors[:, order]
    for i in range(vectors.shape[1]):
        norm = np.linalg.norm(vectors[:, i])
        if norm == 0:
            raise ValueError("Zero eigenvector encountered.")
        vectors[:, i] /= norm
    return values, vectors


def solve_a_matrix(
    nominal: np.ndarray,
    phase: PhaseParameters,
    eigenvectors: np.ndarray,
) -> np.ndarray:
    """Solve A_ji from Eq. (7) using the eigenvector expansion A_ji=N_ji|A_i|."""

    f_alpha = phase.f_alpha
    f_beta = phase.f_beta
    k_alpha = phase.k_alpha
    k_beta = phase.k_beta

    numerator = f_alpha * (1.0 - k_alpha) + f_beta * (1.0 - k_beta)
    denominator = 1.0 - f_alpha * (1.0 - k_alpha) - f_beta * (1.0 - k_beta)
    if np.any(np.abs(denominator) < 1e-14):
        raise ValueError("Eq. (7) denominator is too close to zero.")
    rhs = numerator / denominator * nominal

    magnitudes = np.linalg.solve(eigenvectors, rhs)
    return eigenvectors @ np.diag(magnitudes)


def fourier_coefficients(phase: PhaseParameters, m: int) -> np.ndarray:
    """a_m in Eq. (9), component-wise for alpha/beta partition difference."""

    return (phase.k_beta - phase.k_alpha) * (2.0 / (m * np.pi)) * np.sin(m * np.pi * phase.f_alpha)


def omega_for_mode(b_m: float, eigenvalues: np.ndarray, velocity: float) -> np.ndarray:
    """omega_m in Eq. (2) for every eigenmode."""

    return 0.5 + np.sqrt(0.25 + (b_m * eigenvalues / velocity) ** 2)


def solve_q_tensor(
    phase: PhaseParameters,
    eigenvectors: np.ndarray,
    eigenvalues: np.ndarray,
    a_matrix: np.ndarray,
    velocity: float,
    lamellar_spacing: float,
    fourier_terms: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve Q_ji(m) from Eq. (10) in the diffusion-eigenvector basis.

    The right-hand side is the Fourier decomposition of the chemically
    patterned flux boundary condition. The unknown Q vectors are expanded in
    the same eigenvectors as A and E.
    """

    n = eigenvectors.shape[0]
    q_tensor = np.zeros((fourier_terms, n, n), dtype=float)
    omega_modes = np.zeros((fourier_terms, n), dtype=float)
    b_modes = np.zeros(fourier_terms, dtype=float)

    interface_shift = np.sum(a_matrix, axis=1)
    for idx, m in enumerate(range(1, fourier_terms + 1)):
        b_m = 2.0 * m * np.pi / lamellar_spacing
        omega = omega_for_mode(b_m, eigenvalues, velocity)
        rhs = fourier_coefficients(phase, m) * interface_shift
        mode_magnitudes = np.linalg.solve(eigenvectors, rhs) / omega
        q_tensor[idx, :, :] = eigenvectors @ np.diag(mode_magnitudes)
        omega_modes[idx, :] = omega
        b_modes[idx] = b_m
    return q_tensor, omega_modes, b_modes


def solve_solute_field(
    alloy: AlloySystem,
    conditions: GrowthConditions,
    fourier_terms: int = 1000,
    phase: PhaseParameters | None = None,
) -> SoluteFieldSolution:
    """Build the full analytical solute field for a velocity/spacing pair."""

    used_phase = phase or alloy.phase_for(
        conditions.velocity,
        conditions.interface_temperature or alloy.physical.temperature_eutectic,
    )
    phi = nonequilibrium_phi(conditions.velocity, alloy.physical.diffusion_speed_liquid)
    d_phi = alloy.physical.diffusion_matrix @ np.diag(phi)
    eigenvalues, eigenvectors = eigensystem(d_phi)
    a_matrix = solve_a_matrix(alloy.nominal_composition, used_phase, eigenvectors)
    q_tensor, omega_modes, b_modes = solve_q_tensor(
        used_phase,
        eigenvectors,
        eigenvalues,
        a_matrix,
        conditions.velocity,
        conditions.lamellar_spacing,
        fourier_terms,
    )
    return SoluteFieldSolution(
        components=alloy.components,
        velocity=conditions.velocity,
        lamellar_spacing=conditions.lamellar_spacing,
        phi=phi,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        a_matrix=a_matrix,
        q_tensor=q_tensor,
        omega_modes=omega_modes,
        b_modes=b_modes,
        nominal_composition=alloy.nominal_composition,
        phase=used_phase,
    )
