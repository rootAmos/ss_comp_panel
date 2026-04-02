"""
composite_panel
===============
Classical Laminate Theory toolkit for composite panel analysis.

Designed for aerospace structural sizing – specifically supersonic airframe
skin panels.  Core workflow:

    from composite_panel import PlyMaterial, Ply, Laminate
    from composite_panel import failure, aero_loads

Quick start
-----------
    from composite_panel import Ply, Laminate, IM7_8552

    mat   = IM7_8552()
    plies = [Ply(mat, 0.125e-3, θ) for θ in [-45, 0, 45, 90, 90, 45, 0, -45]]
    lam   = Laminate(plies)
    print(lam.summary())
"""

from .ply      import Ply, PlyMaterial, IM7_8552, T300_5208
from .laminate import Laminate
from .failure  import (check_laminate, tsai_wu, tsai_hill,
                       max_stress, max_strain, hashin, PlyFailureResult)
from .aero_loads import (PanelLoads, WingGeometry, wing_panel_loads,
                          supersonic_panel_loads,
                          elliptic_spanwise_load, ackeret_panel_pressure,
                          hypersonic_panel_pressure, oblique_shock_panel_pressure,
                          panel_pressure)
from .thermal  import (PlyThermal, ThermalState, IM7_8552_thermal,
                        alpha_bar, thermal_resultants,
                        aero_wall_temperature, aero_heat_flux,
                        equilibrium_wall_temperature, thermal_state_from_flight)
from .buckling  import (Nxx_cr, Nyy_cr, Nxy_cr, buckling_rf,
                        Nxx_cr_smooth, Nyy_cr_smooth, Nxy_cr_smooth,
                        buckling_rf_smooth, suggest_mode_number)
from .loads_db    import LoadCase, LoadsDatabase
from .optimizer   import (
    optimize_laminate, optimize_laminate_multicase,
    optimize_wing, pareto_sweep, detect_balance_pairs,
    OptimizationResult, MulticaseOptimizationResult, WingOptimizationResult,
)
from .trim        import TrimState, trim_alpha, trim_table, lift_curve_slope
from .aeroelastic import (AeroelasticResult, static_aeroelastic,
                           wing_bending_stiffness, wing_torsional_stiffness,
                           wing_coupling_stiffness)
from .lamination_parameters import (
    material_invariants,
    lamination_parameters,
    abd_from_lamination_params,
    stiffness_polar,
    plot_lp_feasibility,
    bend_twist_coupling_index,
    shear_extension_coupling_index,
)

__all__ = [
    # ply / laminate
    "Ply", "PlyMaterial", "IM7_8552", "T300_5208",
    "Laminate",
    # failure
    "check_laminate", "tsai_wu", "tsai_hill", "max_stress", "max_strain",
    "hashin", "PlyFailureResult",
    # aero loads
    "PanelLoads", "WingGeometry", "wing_panel_loads",
    "supersonic_panel_loads", "elliptic_spanwise_load", "ackeret_panel_pressure",
    "hypersonic_panel_pressure", "panel_pressure",
    # thermal
    "PlyThermal", "ThermalState", "IM7_8552_thermal",
    "alpha_bar", "thermal_resultants",
    "aero_wall_temperature", "aero_heat_flux",
    "equilibrium_wall_temperature", "thermal_state_from_flight",
    # buckling
    "Nxx_cr", "Nyy_cr", "Nxy_cr", "buckling_rf",
    "Nxx_cr_smooth", "Nyy_cr_smooth", "Nxy_cr_smooth",
    "buckling_rf_smooth", "suggest_mode_number",
    # loads database
    "LoadCase", "LoadsDatabase",
    # optimizer
    "optimize_laminate", "optimize_laminate_multicase",
    "optimize_wing", "pareto_sweep", "detect_balance_pairs",
    "OptimizationResult", "MulticaseOptimizationResult", "WingOptimizationResult",
    # trim / flight dynamics
    "TrimState", "trim_alpha", "trim_table", "lift_curve_slope",
    # aeroelastic
    "AeroelasticResult", "static_aeroelastic",
    "wing_bending_stiffness", "wing_torsional_stiffness", "wing_coupling_stiffness",
    # lamination parameters
    "material_invariants", "lamination_parameters", "abd_from_lamination_params",
    "stiffness_polar", "plot_lp_feasibility",
    "bend_twist_coupling_index", "shear_extension_coupling_index",
]

__version__ = "0.1.0"
