from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .solute_field import SoluteFieldSolution
from .workflow import WorkflowResult


def ensure_directory(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def write_interface_profile(
    solute: SoluteFieldSolution,
    path: str | Path,
    n_points: int = 1001,
    periods: float = 2.0,
) -> None:
    x, c = solute.interface_profile(n_points=n_points, periods=periods)
    with Path(path).open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["x_m", *[f"C_{name}" for name in solute.components]])
        for idx in range(x.size):
            writer.writerow([f"{x[idx]:.12e}", *[f"{c[j, idx]:.12e}" for j in range(len(solute.components))]])


def write_z_profile(
    solute: SoluteFieldSolution,
    path: str | Path,
    z_max: float,
    n_points: int = 1001,
    x: float = 0.0,
) -> None:
    z, c = solute.z_profile(z_max=z_max, n_points=n_points, x=x)
    with Path(path).open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["z_m", *[f"C_{name}" for name in solute.components]])
        for idx in range(z.size):
            writer.writerow([f"{z[idx]:.12e}", *[f"{c[j, idx]:.12e}" for j in range(len(solute.components))]])


def write_lambda_search(result: WorkflowResult, path: str | Path) -> None:
    lambdas = result.lambda_result.searched_lambdas
    undercooling = result.lambda_result.searched_total_undercooling
    with Path(path).open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["lambda_m", "total_undercooling_K"])
        for lam, dt in zip(lambdas, undercooling):
            writer.writerow([f"{lam:.12e}", f"{dt:.12e}"])


def write_stability_scan(result: WorkflowResult, path: str | Path) -> None:
    stability = result.stability_result
    with Path(path).open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["q_1_per_m", "growth_rate"])
        for q, rate in zip(stability.q_values, stability.growth_rates):
            writer.writerow([f"{q:.12e}", f"{rate:.12e}" if np.isfinite(rate) else "nan"])


def write_summary(results: list[WorkflowResult], path: str | Path) -> None:
    with Path(path).open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "velocity_m_per_s",
                "interface_temperature_K",
                "lambda_m",
                "total_undercooling_K",
                "constitutional_dendritic_K",
                "constitutional_eutectic_K",
                "curvature_K",
                "kinetic_K",
                "max_growth_rate",
                "q_at_max_1_per_m",
                "stable",
            ]
        )
        for result in results:
            undercooling = result.lambda_result.undercooling
            stability = result.stability_result
            writer.writerow(
                [
                    f"{result.velocity:.12e}",
                    f"{result.interface_temperature:.12e}",
                    f"{result.lamellar_spacing:.12e}",
                    f"{undercooling.total:.12e}",
                    f"{undercooling.constitutional_dendritic:.12e}",
                    f"{undercooling.constitutional_eutectic:.12e}",
                    f"{undercooling.curvature:.12e}",
                    f"{undercooling.kinetic:.12e}",
                    f"{stability.max_growth_rate:.12e}",
                    f"{stability.q_at_max:.12e}",
                    str(result.stable),
                ]
            )
