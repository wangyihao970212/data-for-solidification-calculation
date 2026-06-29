# Growth-Stability Framework for Complex-Component Eutectics

This folder reorganizes the earlier draft scripts into a reusable Python implementation of the manuscript equations. It is intentionally split into small modules so the solute field, undercooling loop, interfacial response, and marginal-stability calculation can be checked independently.

## Files

- `growth_stability/parameters.py`: alloy, phase, physical, and numerical input dataclasses.
- `growth_stability/solute_field.py`: Eqs. (1)-(10), including `phi_i`, eigenvalues `B_i`, eigenvectors `N_ji`, `A_ji`, Fourier coefficients, `omega_mi`, `Q_ji(m)`, and `C_j(x,z)`.
- `growth_stability/undercooling.py`: Eqs. (19)-(27), minimum-undercooling search, and the Fig. 4 interface-temperature iteration.
- `growth_stability/stability.py`: perturbation stability, including the Hunziker-style solution of `E_ji` and the marginal-stability function.
- `growth_stability/interface_response.py`: optional partial-drag/chemical-potential equations. It accepts user-provided Gibbs-energy or chemical-potential functions and does not require SciPy.
- `growth_stability/workflow.py`: complete velocity loop joining solute field, lambda selection, temperature iteration, and stability.
- `growth_stability/io_utils.py`: CSV export helpers.
- `examples/system_template.py`: user-editable system template with no alloy-specific values filled in.
- `examples/run_demo.py`: runnable example that generates concentration profiles, lambda search data, stability scans, and summary CSV files.

## Run

From this directory:

```powershell
python examples\run_demo.py
```

Before running, fill `examples/system_template.py` with the parameters for your own alloy system. The results are written to `outputs/`.

For the full workflow entry point:

```powershell
python run_full.py
```

## Notes for User Systems

The code keeps the full multicomponent eigenvector form rather than the simplified diagonal-only form. No alloy-specific thermodynamic database or parameter set is bundled with this public template. Users should provide their own phase fractions, partition coefficients, diffusion matrix, liquidus slopes, Gibbs-Thomson coefficients, thermal parameters, tie-line derivative matrix, and optional interfacial kinetic data.

`interface_response.py` is optional in the current template because users may either directly supply phase fractions and partition coefficients or update them through a partial-drag model. If the partial-drag model is used to update `k_alpha` and `k_beta`, provide chemical-potential functions and connect the resulting phase update through `InterfaceKinetics.update_phase_parameters`. For constrained multicomponent calculations, one independent solid or liquid solute can be fixed through `fixed_solid_solutes` or `fixed_liquid_solutes`.
