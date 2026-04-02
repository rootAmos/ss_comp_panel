"""
composite_panel.aeroelastic
---------------------------
1-D static aeroelastic solution for a tapered swept wing.

Iterative Euler-Bernoulli approach (root-clamped cantilever):
    1. rigid loads at alpha_0 via wing_panel_loads()
    2. EI(y) from CLT A-matrix: E_eff=A11/h, EI=2*E_eff*b_box*(h_box/2)^2
    3. bending slope theta(y) = integral(M/EI)
    4. washout: delta_alpha(y) = -theta(y)*tan(sweep)
    5. alpha_eff = alpha_0 + delta_alpha, repeat until converged

Sign: upward theta on a swept wing → nose-down twist → delta_alpha < 0 (relief).

Ref: Bisplinghoff, Ashley & Halfman — Aeroelasticity (Dover, 1955)
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as _np

try:
    from .aero_loads import WingGeometry, wing_panel_loads, PanelLoads, _isa as isa_atmosphere
except ImportError:
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
    from composite_panel.aero_loads import WingGeometry, wing_panel_loads, PanelLoads, _isa as isa_atmosphere




@dataclass
class AeroelasticResult:
    """
    Output of static_aeroelastic().

    alpha_eff = alpha_rigid + delta_alpha(eta) [deg]
    delta_alpha < 0 → washout (swept bending, nose-down tip twist)
    aeroelastic_relief = (Nyy_elastic - Nyy_rigid)/|Nyy_rigid| at tip;
        Nyy < 0, so positive = tip unloaded
    """
    etas:              _np.ndarray
    alpha_rigid:       float
    alpha_eff:         _np.ndarray
    delta_alpha:       _np.ndarray
    tip_deflection:    float
    bending_slope:     _np.ndarray
    EI:                _np.ndarray
    loads_rigid:       List[PanelLoads]
    loads_elastic:     List[PanelLoads]
    n_iterations:      int
    converged:         bool
    aeroelastic_relief: float

    def summary(self) -> str:
        sign = "↓ tip loads REDUCED (washout)" if self.aeroelastic_relief > 0 else "↑ tip loads INCREASED (divergence)"
        return (
            f"Aeroelastic result:\n"
            f"  Rigid AoA        : {self.alpha_rigid:.2f}°\n"
            f"  Elastic AoA tip  : {self.alpha_eff[-1]:.2f}°  "
            f"(Δα = {self.delta_alpha[-1]:+.2f}°)\n"
            f"  Tip deflection   : {self.tip_deflection*100:.1f} cm\n"
            f"  Aeroelastic relief: {self.aeroelastic_relief*100:+.1f}%  {sign}\n"
            f"  Iterations       : {self.n_iterations}  "
            f"({'converged' if self.converged else 'NOT CONVERGED'})"
        )




def wing_bending_stiffness(
    wing:            WingGeometry,
    laminate_A11:    float,
    laminate_h:      float,
    etas:            _np.ndarray,
    box_chord_fraction: float = 0.50,
) -> _np.ndarray:
    """
    Spanwise EI(y) from CLT A-matrix, thin-walled rectangular box model.

      E_eff = A11 / h,   EI = 2 * E_eff * b_box * (h_box/2)^2
      b_box = box_chord_fraction * chord(eta),  h_box = t_over_c * chord(eta)

    laminate_A11 = Laminate.A[0,0] [N/m]
    """
    EI = _np.zeros(len(etas))
    for i, eta in enumerate(etas):
        c   = wing.chord(eta)
        b_b = box_chord_fraction * c
        h_b = wing.t_over_c * c
        E_eff = laminate_A11 / max(laminate_h, 1e-6)
        EI[i] = 2.0 * E_eff * b_b * (h_b / 2.0) ** 2
    return EI




def euler_bernoulli_cantilever(
    y:       _np.ndarray,
    EI:      _np.ndarray,
    q_dist:  _np.ndarray,
) -> tuple:
    """
    Compute bending slope and deflection for a cantilever with
    variable EI and distributed transverse load q(y).

    Euler-Bernoulli cantilever (clamped at y=0, free at y=b):

        d²M/dy² = q(y)         →  M(y) = ∫_y^b q(η)(η-y)dη
        d²w/dy² = M(y)/EI(y)   →  θ(y) = ∫_0^y M(ξ)/EI(ξ) dξ
        dw/dy   = θ(y)         →  w(y)  = ∫_0^y θ(ξ) dξ

    Boundary conditions: w(0) = 0  (no root deflection — wing root clamped)
                         θ(0) = 0  (no root slope)

    Parameters
    ----------
    y       : spanwise stations [m]  (sorted root to tip, y[0]=0)
    EI      : bending stiffness [N·m²] at each station
    q_dist  : distributed upward load [N/m] at each station (lift per span)

    Returns
    -------
    theta   : bending slope dw/dy [rad] at each station
    w       : vertical deflection [m] at each station
    M_beam  : bending moment [N·m] at each station
    """
    n = len(y)
    M_beam = _np.zeros(n)
    theta  = _np.zeros(n)
    w      = _np.zeros(n)

    # M(y) = integral_y^b q(eta)*(eta-y) deta
    for i in range(n):
        eta_      = y[i:]
        M_beam[i] = _np.trapezoid(q_dist[i:] * (eta_ - y[i]), eta_)

    kappa = M_beam / _np.maximum(EI, 1e-6)

    for i in range(1, n):
        theta[i] = theta[i-1] + _np.trapezoid(kappa[i-1:i+1], y[i-1:i+1])

    for i in range(1, n):
        w[i] = w[i-1] + _np.trapezoid(theta[i-1:i+1], y[i-1:i+1])

    return theta, w, M_beam




def static_aeroelastic(
    wing:           WingGeometry,
    mach:           float,
    altitude_m:     float,
    alpha_rigid_deg: float,
    n_load:         float,
    laminate_A11:   float,
    laminate_h:     float,
    n_stations:     int   = 20,
    box_fraction:   float = 0.50,
    max_iter:       int   = 10,
    tol_deg:        float = 0.01,
) -> AeroelasticResult:
    """
    Iterative static aeroelastic solution.  Converges when max spanwise
    |delta_alpha| change < tol_deg.  A11/h assumed spanwise-constant.
    """
    etas = _np.linspace(0.02, 0.98, n_stations)
    y    = etas * wing.semi_span        # physical span coordinate [m]

    sweep_rad  = math.radians(wing.sweep_le_deg)
    tan_sweep  = math.tan(sweep_rad)

    # ── Atmosphere ────────────────────────────────────────────────────────────
    rho, a_sound = isa_atmosphere(altitude_m)
    q_inf        = 0.5 * rho * (mach * a_sound) ** 2

    EI = wing_bending_stiffness(wing, laminate_A11, laminate_h, etas, box_fraction)

    alpha_eff   = _np.full(n_stations, alpha_rigid_deg)
    loads_rigid = [
        wing_panel_loads(wing, eta, mach, altitude_m, alpha_rigid_deg, n_load)
        for eta in etas
    ]

    n_iter    = 0
    converged = False

    for iteration in range(max_iter):
        loads = [
            wing_panel_loads(wing, etas[i], mach, altitude_m,
                             float(alpha_eff[i]), n_load)
            for i in range(n_stations)
        ]

        # Nyy = -dP*chord/2  →  dP = -2*Nyy/chord
        q_lift = _np.array([
            -2.0 * loads[i].Nyy / max(wing.chord(etas[i]), 1e-3)
            for i in range(n_stations)
        ])

        theta, w_deflect, _ = euler_bernoulli_cantilever(y, EI, q_lift)

        delta_alpha_deg = _np.degrees(-theta * tan_sweep)   # washout: -theta*tan(sweep)
        alpha_new       = alpha_rigid_deg + delta_alpha_deg

        max_change = float(_np.max(_np.abs(alpha_new - alpha_eff)))
        alpha_eff  = alpha_new
        n_iter     = iteration + 1

        if max_change < tol_deg:
            converged = True
            break

    loads_elastic = [
        wing_panel_loads(wing, etas[i], mach, altitude_m,
                         float(alpha_eff[i]), n_load)
        for i in range(n_stations)
    ]

    # Nxx at tip = 0 (free-end BCs), use Nyy for relief metric
    Nyy_rigid_tip   = loads_rigid[-1].Nyy
    Nyy_elastic_tip = loads_elastic[-1].Nyy
    if abs(Nyy_rigid_tip) > 1.0:
        relief = (Nyy_elastic_tip - Nyy_rigid_tip) / abs(Nyy_rigid_tip)
    else:
        relief = 0.0

    return AeroelasticResult(
        etas              = etas,
        alpha_rigid       = alpha_rigid_deg,
        alpha_eff         = alpha_eff,
        delta_alpha       = delta_alpha_deg,
        tip_deflection    = float(w_deflect[-1]),
        bending_slope     = theta,
        EI                = EI,
        loads_rigid       = loads_rigid,
        loads_elastic     = loads_elastic,
        n_iterations      = n_iter,
        converged         = converged,
        aeroelastic_relief= relief,
    )


if __name__ == "__main__":
    import sys as _sys
    _sys.stdout.reconfigure(encoding="utf-8")
    wing = WingGeometry(
        semi_span    = 4.5,     # m  — half-span
        root_chord   = 2.0,     # m
        taper_ratio  = 0.3,     # tip/root chord
        sweep_le_deg = 45.0,    # deg  leading-edge sweep
        t_over_c     = 0.04,    # structural box depth fraction
        mtow_n       = 150_000.0,  # N  max take-off weight
    )

    # IM7/8552 [0/45/-45/90]s 8-ply laminate (h ~ 1 mm)
    # Ex ~ 52 GPa  →  A11 = Ex * h = 52e9 * 1e-3 = 52e6 N/m
    laminate_A11 = 52e6   # N/m
    laminate_h   = 1.0e-3  # m

    result = static_aeroelastic(
        wing             = wing,
        mach             = 1.7,
        altitude_m       = 15_000.0,
        alpha_rigid_deg  = 3.5,
        n_load           = 2.5,
        laminate_A11     = laminate_A11,
        laminate_h       = laminate_h,
        n_stations       = 20,
        box_fraction     = 0.50,
    )
    print(result.summary())
