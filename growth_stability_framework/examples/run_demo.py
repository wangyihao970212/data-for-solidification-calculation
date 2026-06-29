from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.system_template import build_system, numerical_settings, velocity_grid
from growth_stability.io_utils import (
    ensure_directory,
    write_interface_profile,
    write_lambda_search,
    write_stability_scan,
    write_summary,
    write_z_profile,
)
from growth_stability.workflow import find_transition_velocity, scan_velocity_range


def main() -> None:
    try:
        system = build_system()
        settings = numerical_settings()
        velocities = velocity_grid()
    except NotImplementedError as exc:
        print(f"Input required: {exc}")
        return
    results = scan_velocity_range(system, velocities, settings=settings)

    output_dir = ensure_directory(ROOT / "outputs")
    write_summary(results, output_dir / "summary.csv")
    transition = find_transition_velocity(results)
    for result in results:
        label = f"V_{result.velocity:.2e}".replace("+", "")
        write_interface_profile(
            result.lambda_result.solute_solution,
            output_dir / f"{label}_interface_profile.csv",
            n_points=1001,
            periods=2.0,
        )
        write_z_profile(
            result.lambda_result.solute_solution,
            output_dir / f"{label}_z_profile.csv",
            z_max=5.0e-5,
            n_points=1001,
            x=0.0,
        )
        write_lambda_search(result, output_dir / f"{label}_lambda_search.csv")
        write_stability_scan(result, output_dir / f"{label}_stability_scan.csv")

    print(f"Output directory: {output_dir}")
    for result in results:
        print(
            "V={:.3e} m/s, Ti={:.3f} K, lambda={:.3e} m, "
            "max_growth={:.3e}, stable={}".format(
                result.velocity,
                result.interface_temperature,
                result.lamellar_spacing,
                result.stability_result.max_growth_rate,
                result.stable,
            )
        )
    if transition is not None:
        print("Transition bracket: {:.3e} to {:.3e} m/s".format(*transition))
    else:
        print("No stability transition was bracketed in the demo velocity list.")


if __name__ == "__main__":
    main()
