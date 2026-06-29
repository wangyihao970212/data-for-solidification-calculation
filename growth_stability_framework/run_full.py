from __future__ import annotations

from pathlib import Path

from examples.system_template import build_system, numerical_settings, velocity_grid
from growth_stability import scan_velocity_range, find_transition_velocity
from growth_stability.io_utils import (
    ensure_directory,
    write_interface_profile,
    write_lambda_search,
    write_stability_scan,
    write_summary,
    write_z_profile,
)


ROOT = Path(__file__).resolve().parent


def main() -> None:
    try:
        system = build_system()
        settings = numerical_settings()
        velocities = velocity_grid()
    except NotImplementedError as exc:
        print(f"Input required: {exc}")
        return

    output_dir = ensure_directory(ROOT / "outputs_full")
    results = scan_velocity_range(system, velocities, settings=settings)
    write_summary(results, output_dir / "summary.csv")

    for result in results:
        label = f"V_{result.velocity:.2e}".replace("+", "")
        write_interface_profile(
            result.lambda_result.solute_solution,
            output_dir / f"{label}_interface_profile.csv",
            n_points=2001,
            periods=2.0,
        )
        write_z_profile(
            result.lambda_result.solute_solution,
            output_dir / f"{label}_z_profile.csv",
            z_max=5.0e-5,
            n_points=2001,
            x=0.0,
        )
        write_lambda_search(result, output_dir / f"{label}_lambda_search.csv")
        write_stability_scan(result, output_dir / f"{label}_stability_scan.csv")

    transition = find_transition_velocity(results)
    print(f"Output directory: {output_dir}")
    for result in results:
        print(
            "V={:.3e} m/s, Ti={:.4f} K, lambda={:.4e} m, "
            "max_growth={:.4e}, q_at_max={:.4e}, stable={}".format(
                result.velocity,
                result.interface_temperature,
                result.lamellar_spacing,
                result.stability_result.max_growth_rate,
                result.stability_result.q_at_max,
                result.stable,
            )
        )
    if transition is None:
        print("No stability transition was bracketed in this velocity list.")
    else:
        print("Transition bracket: {:.4e} to {:.4e} m/s".format(*transition))


if __name__ == "__main__":
    main()
