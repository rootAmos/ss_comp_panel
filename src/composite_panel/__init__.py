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
                       max_stress, max_strain, PlyFailureResult)
from .aero_loads import (PanelLoads, supersonic_panel_loads,
                          elliptic_spanwise_load, ackeret_panel_pressure)

__all__ = [
    "Ply", "PlyMaterial", "IM7_8552", "T300_5208",
    "Laminate",
    "check_laminate", "tsai_wu", "tsai_hill", "max_stress", "max_strain",
    "PlyFailureResult",
    "PanelLoads", "supersonic_panel_loads",
    "elliptic_spanwise_load", "ackeret_panel_pressure",
]

__version__ = "0.1.0"
