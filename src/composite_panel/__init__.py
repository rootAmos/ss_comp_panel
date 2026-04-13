"""
composite_panel
===============
Classical Laminate Theory toolkit for composite panel analysis.

Designed for aerospace structural sizing of high-speed airframe skin panels.
Primary workflow is to import external running loads, then size the laminate
against those inputs. Closed-form aero helpers remain for screening/demo use.

Core workflow:

    from composite_panel import PlyMaterial, Ply, Laminate
    from composite_panel import failure, aero_loads

Quick start
-----------
    from composite_panel import Ply, Laminate, IM7_8552

    mat   = IM7_8552()
    plies = [Ply(mat, 0.125e-3, theta) for theta in [-45, 0, 45, 90, 90, 45, 0, -45]]
    lam   = Laminate(plies)
    print(lam.summary())
"""

from .ply      import Ply, PlyMaterial, IM7_8552, T300_5208
from composite_panel.laminate import Laminate
from .failure  import (check_laminate, tsai_wu, tsai_hill,
                       max_stress, max_strain, hashin, PlyFailureResult)
from composite_panel.aero_loads import (PanelLoads, WingGeometry, wing_panel_loads,
                          ackeret_panel_pressure,
                          hypersonic_panel_pressure,
                          panel_pressure)
from .thermal  import (PlyThermal, ThermalState, IM7_8552_thermal,
                        alpha_bar, thermal_resultants,
                        aero_wall_temperature, aero_heat_flux,
                        equilibrium_wall_temperature, thermal_state_from_flight)
from .buckling  import (Nxx_cr, Nyy_cr, Nxy_cr, buckling_rf,
                        Nxx_cr_smooth, Nyy_cr_smooth, Nxy_cr_smooth,
                        buckling_rf_smooth, suggest_mode_number)
from .loads_db    import LoadCase, LoadsDatabase
from .arrow_wing  import (ArrowWingGeometry, arrow_wing_panel_loads,
                           arrow_wing_loads_database)
from .optimizer   import (
    optimize_laminate, optimize_laminate_multicase,
    optimize_wing, detect_balance_pairs,
    optimize_laminate_aeroelastic,
    OptimizationResult, MulticaseOptimizationResult, WingOptimizationResult,
    AeroelasticOptimizationResult,
)
from composite_panel.aeroelastic import (AeroelasticResult, static_aeroelastic,
                           wing_bending_stiffness, wing_torsional_stiffness,
                           wing_coupling_stiffness)
__all__ = [
    # ply / laminate
    "Ply", "PlyMaterial", "IM7_8552", "T300_5208",
    "Laminate",
    # failure
    "check_laminate", "tsai_wu", "tsai_hill", "max_stress", "max_strain",
    "hashin", "PlyFailureResult",
    # aero loads
    "PanelLoads", "WingGeometry", "wing_panel_loads",
    "ackeret_panel_pressure", "hypersonic_panel_pressure",
    "panel_pressure",
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
    # arrow wing (CR-132575)
    "ArrowWingGeometry", "arrow_wing_panel_loads", "arrow_wing_loads_database",
    # optimizer
    "optimize_laminate", "optimize_laminate_multicase",
    "optimize_wing", "detect_balance_pairs",
    "optimize_laminate_aeroelastic",
    "OptimizationResult", "MulticaseOptimizationResult", "WingOptimizationResult",
    "AeroelasticOptimizationResult",
    # aeroelastic
    "AeroelasticResult", "static_aeroelastic",
    "wing_bending_stiffness", "wing_torsional_stiffness", "wing_coupling_stiffness",
]

__version__ = "0.1.0"
