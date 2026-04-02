"""
composite_panel.thermal
-----------------------
Aerodynamic heating model and CLT thermal load resultants.

BACKGROUND
==========
Hypersonic skin panels experience two superimposed sources of thermal load:

  1. Cure-residual stresses
     IM7/8552 is cured at ~177 C.  When the structure cools to operating
     temperature, the mismatch between the high alpha_2 (transverse) and
     near-zero alpha_1 (fibre) CTEs locks in residual stress.  For a
     [0/90]s laminate this can exceed 50 MPa — comparable to the applied
     mechanical loads at low Mach.

  2. Aerodynamic heating
     At Mach 4+, the adiabatic wall temperature exceeds 600 C.  The skin
     temperature (T_wall) depends on the TPS design and cooling strategy.
     Even with a ceramic TPS, the underlying composite may see 150-300 C
     above cure temperature, reversing the residual stress sign and adding
     significant through-thickness gradients.

CLT THERMAL FORMULATION
========================
The thermal load resultants N_T and M_T are added to the mechanical loads
before inverting the ABD compliance:

    [eps0, kappa] = abd @ ( [N_mech + N_T, M_mech + M_T] )

Ply principal-axis CTE vector:   alpha_12 = [alpha_1, alpha_2, 0]
Transformed to laminate axes:
    alpha_x  = alpha_1*c^2 + alpha_2*s^2
    alpha_y  = alpha_1*s^2 + alpha_2*c^2
    alpha_xy = 2*(alpha_1 - alpha_2)*c*s      (engineering shear CTE)

Thermal resultants (exact, integrated per ply):
    N_T  = sum_k  Q_bar_k @ alpha_bar_k  *  delta_T_k  *  t_k       [N/m]
    M_T  = sum_k  Q_bar_k @ alpha_bar_k  *  delta_T_k  *  z_mid_k * t_k  [N.m/m]

where delta_T_k = T_wall_k - T_cure is the temperature rise at ply k's
mid-plane.  For a uniform temperature field delta_T_k = const = delta_T.
For a through-thickness gradient, delta_T varies linearly.

AERODYNAMIC HEATING
===================
Eckert reference-temperature method (1955) for turbulent flat-plate flow:

    T_aw  = T_inf * (1 + r*(gamma-1)/2 * M^2)       adiabatic wall temp
    T*    = 0.5*(T_inf + T_wall) + 0.22*(T_aw - T_inf)   reference temp
    St*   = 0.0296 * Re_x*^(-0.2) * Pr*^(-2/3)          Stanton number
    q_dot = St* * rho* * V * cp* * (T_aw - T_wall)       heat flux [W/m^2]

For structural sizing: T_wall is an input (determined by TPS/cooling design).
The adiabatic wall temperature gives the maximum possible skin temperature.

References
----------
Kassapoglou, C. — Design and Analysis of Composite Structures (2013) Ch. 7
Jones, R.M.     — Mechanics of Composite Materials (1999) Ch. 5
Eckert, E.R.G.  — Engineering Relations for Friction and Heat Transfer (1955)
ESDU 77028      — Buckling of flat orthotropic plates under combined loads
"""

from __future__ import annotations

import numpy as _np
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Union

try:
    from .ply import PlyMaterial, Ply
except ImportError:
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
    from composite_panel.ply import PlyMaterial, Ply


# ---------------------------------------------------------------------------
# Ply thermal properties
# ---------------------------------------------------------------------------

@dataclass
class PlyThermal:
    """
    Thermal expansion coefficients for a unidirectional ply.

    IM7/8552 typical values (Hexcel datasheet + Bowles & Tompkins 1989):
      alpha_1 = 0.3e-6 /K   (fibre-dominated — near zero, slightly positive)
      alpha_2 = 28.8e-6 /K  (matrix-dominated — large positive)

    The large alpha_1 / alpha_2 mismatch is the root cause of cure-residual
    stresses and the thermal coupling sensitivity seen in cross-ply laminates.

    Parameters
    ----------
    alpha_1 : float
        CTE along fibre direction [1/K]
    alpha_2 : float
        CTE transverse to fibre [1/K]
    T_cure : float
        Laminate cure temperature [K] (stress-free reference state)
    """
    alpha_1: float
    alpha_2: float
    T_cure:  float = 450.15   # 177 C  — standard IM7/8552 autoclave cure cycle


def IM7_8552_thermal() -> PlyThermal:
    """
    Thermal properties for Hexcel IM7/8552 UD carbon/epoxy.

    Source: Bowles & Tompkins (1989), Hexcel HexPly 8552 datasheet.
    Cure temperature: 177 C (450 K) — standard autoclave cycle.
    """
    return PlyThermal(
        alpha_1 = 0.3e-6,    # /K  fibre direction  (near-zero — carbon dominated)
        alpha_2 = 28.8e-6,   # /K  transverse       (matrix dominated)
        T_cure  = 450.15,    # K   177 C autoclave cure
    )


# ---------------------------------------------------------------------------
# Transformed CTE vector
# ---------------------------------------------------------------------------

def alpha_bar(pt: PlyThermal, angle_rad: float) -> _np.ndarray:
    """
    CTE vector transformed to laminate (x-y) axes.

    From ply principal axes [alpha_1, alpha_2, 0] to laminate axes via the
    inverse of the strain transformation (Jones 1999, eq. 5.25):

        alpha_x  = alpha_1*c^2 + alpha_2*s^2
        alpha_y  = alpha_1*s^2 + alpha_2*c^2
        alpha_xy = 2*(alpha_1 - alpha_2)*c*s     (engineering shear CTE)

    Parameters
    ----------
    pt        : PlyThermal
    angle_rad : ply fibre angle [rad]

    Returns
    -------
    (3,) ndarray  [alpha_x, alpha_y, alpha_xy]
    """
    a1, a2 = pt.alpha_1, pt.alpha_2
    c = _np.cos(angle_rad)
    s = _np.sin(angle_rad)
    return _np.array([
        a1 * c**2 + a2 * s**2,
        a1 * s**2 + a2 * c**2,
        2.0 * (a1 - a2) * c * s,
    ])


# ---------------------------------------------------------------------------
# Thermal load resultants
# ---------------------------------------------------------------------------

@dataclass
class ThermalState:
    """
    Temperature state for a laminate.

    Parameters
    ----------
    T_wall_inner : float
        Temperature at the inner (structural) skin surface [K].
        Equals T_outer for uniform temperature; lower than T_outer when
        there is a through-thickness gradient from aerodynamic heating.
    T_wall_outer : float
        Temperature at the outer (aero) skin surface [K].
        For a TPS-protected skin, this is the temperature behind the TPS.
    T_cure : float
        Cure temperature [K].  Thermal loads are zero at this temperature.
    """
    T_wall_outer: float    # K — hot face (aero side)
    T_wall_inner: float    # K — cold face (structural side / bond line)
    T_cure:       float    # K — stress-free reference

    @property
    def delta_T_mean(self) -> float:
        """Mean temperature rise from cure [K]."""
        return 0.5 * (self.T_wall_outer + self.T_wall_inner) - self.T_cure

    @property
    def delta_T_gradient(self) -> float:
        """Through-thickness temperature gradient [K] (outer - inner)."""
        return self.T_wall_outer - self.T_wall_inner

    def delta_T_at_z(self, z: float, h: float) -> float:
        """
        Temperature rise above cure at through-thickness coordinate z [m].
        Linear interpolation between inner and outer face temperatures.
        z in [-h/2, +h/2], z = +h/2 is outer (hot) face.
        """
        T_z = self.T_wall_inner + (z / h + 0.5) * self.delta_T_gradient
        return T_z - self.T_cure


def thermal_resultants(
    plies:          List[Ply],
    ply_thermals:   List[PlyThermal],
    thermal_state:  ThermalState,
    z_interfaces:   _np.ndarray,
) -> tuple:
    """
    Compute CLT thermal force and moment resultants N_T and M_T.

    N_T = sum_k  Q_bar_k @ alpha_bar_k  *  delta_T_k  *  t_k
    M_T = sum_k  Q_bar_k @ alpha_bar_k  *  delta_T_k  *  z_mid_k * t_k

    For a uniform delta_T (no through-thickness gradient), N_T is non-zero
    for any laminate (drives in-plane expansion/contraction) but M_T is zero
    for symmetric laminates (odd-moment integral vanishes).

    For a through-thickness gradient, M_T is non-zero even for symmetric
    laminates — this is the thermally-induced curvature (bimetal-strip effect).

    Parameters
    ----------
    plies         : list of Ply objects
    ply_thermals  : list of PlyThermal, one per ply (same order as plies)
    thermal_state : ThermalState defining T_wall and T_cure
    z_interfaces  : (n_plies+1,) array of ply interface z-coordinates [m]

    Returns
    -------
    N_T : (3,) ndarray  [N_Tx, N_Ty, N_Txy]  [N/m]
    M_T : (3,) ndarray  [M_Tx, M_Ty, M_Txy]  [N.m/m]
    """
    if len(ply_thermals) != len(plies):
        raise ValueError(
            f"ply_thermals length ({len(ply_thermals)}) must match "
            f"plies length ({len(plies)})"
        )

    h_total = z_interfaces[-1] - z_interfaces[0]

    N_T = _np.zeros(3)
    M_T = _np.zeros(3)

    for k, (ply, pt) in enumerate(zip(plies, ply_thermals)):
        z0    = z_interfaces[k]
        z1    = z_interfaces[k + 1]
        z_mid = 0.5 * (z0 + z1)
        t_k   = z1 - z0

        # Temperature at ply mid-plane
        delta_T_k = thermal_state.delta_T_at_z(z_mid, h_total)

        # Transformed CTE in laminate axes
        ab = alpha_bar(pt, _np.radians(ply.angle_deg))   # (3,)

        # Rotated stiffness * CTE product
        Qb_ab = ply.Q_bar @ ab    # (3,)

        N_T += Qb_ab * delta_T_k * t_k
        M_T += Qb_ab * delta_T_k * z_mid * t_k

    return N_T, M_T


# ---------------------------------------------------------------------------
# Aerodynamic heating — Eckert reference-temperature method
# ---------------------------------------------------------------------------

def aero_wall_temperature(
    mach:            float,
    altitude_m:      float,
    recovery_factor: float = 0.89,
) -> float:
    """
    Adiabatic wall temperature from Eckert reference-temperature method.

    The adiabatic wall temperature T_aw is the equilibrium temperature of
    an insulated surface — the maximum skin temperature without active cooling.

    T_aw = T_inf * (1 + r * (gamma-1)/2 * M^2)

    where r = recovery factor:
      Laminar flow  : r = Pr^(1/2) ≈ 0.85
      Turbulent flow: r = Pr^(1/3) ≈ 0.89   (default)

    Parameters
    ----------
    mach            : free-stream Mach number
    altitude_m      : altitude [m]  (ISA atmosphere used)
    recovery_factor : Prandtl recovery factor (default 0.89 turbulent)

    Returns
    -------
    T_aw : float  adiabatic wall temperature [K]
    """
    import aerosandbox as asb
    atm   = asb.Atmosphere(altitude=altitude_m)
    T_inf = atm.temperature()
    gamma = 1.4

    T_aw = T_inf * (1.0 + recovery_factor * (gamma - 1.0) / 2.0 * mach**2)
    return float(T_aw)


def aero_heat_flux(
    mach:       float,
    altitude_m: float,
    x_station:  float,
    T_wall:     float,
    recovery_factor: float = 0.89,
) -> float:
    """
    Convective heat flux at a flat-plate station x [m] from leading edge.

    Uses Eckert's reference temperature method for turbulent flat-plate flow:
      q_dot = St* * rho* * V * cp* * (T_aw - T_wall)     [W/m^2]

    where starred (*) quantities are evaluated at the reference temperature:
      T* = 0.5*(T_inf + T_wall) + 0.22*(T_aw - T_inf)

    Parameters
    ----------
    mach       : free-stream Mach number
    altitude_m : altitude [m]
    x_station  : distance from leading edge [m]  (for Reynolds number)
    T_wall     : current wall temperature [K]
    recovery_factor : Prandtl recovery factor

    Returns
    -------
    q_dot : float  heat flux [W/m^2]  (positive = into structure)
    """
    import aerosandbox as asb

    atm   = asb.Atmosphere(altitude=altitude_m)
    T_inf = float(atm.temperature())
    P_inf = float(atm.pressure())
    V_inf = mach * float(atm.speed_of_sound())
    gamma = 1.4
    R_air = 287.05    # J/(kg.K)
    Pr    = 0.71      # Prandtl number (air, approximately constant)

    T_aw  = aero_wall_temperature(mach, altitude_m, recovery_factor)

    # Reference temperature (Eckert 1955)
    T_star = 0.5 * (T_inf + T_wall) + 0.22 * (T_aw - T_inf)

    # Properties at reference temperature
    rho_star = P_inf / (R_air * T_star)
    mu_star  = 1.458e-6 * T_star**1.5 / (T_star + 110.4)  # Sutherland's law
    cp_star  = gamma * R_air / (gamma - 1.0)               # ~1005 J/(kg.K) for air
    k_star   = mu_star * cp_star / Pr

    # Reynolds number at x using reference conditions
    Re_x_star = rho_star * V_inf * x_station / mu_star

    if Re_x_star < 1.0:
        return 0.0

    # Stanton number — turbulent flat plate (Eckert 1955)
    St_star = 0.0296 * Re_x_star**(-0.2) * Pr**(-2.0 / 3.0)

    q_dot = St_star * rho_star * V_inf * cp_star * (T_aw - T_wall)
    return float(q_dot)


def equilibrium_wall_temperature(
    mach:       float,
    altitude_m: float,
    x_station:  float,
    emissivity: float = 0.85,
    recovery_factor: float = 0.89,
    tol:        float = 1.0,
    max_iter:   int   = 50,
) -> float:
    """
    Radiation-equilibrium wall temperature [K].

    Balances aerodynamic heat flux with thermal radiation:
        q_conv(T_wall) = epsilon * sigma * T_wall^4

    This is the steady-state skin temperature for a radiatively cooled
    surface — appropriate for ceramic TPS or unprotected metallic skins.
    Converged by bisection.

    Parameters
    ----------
    mach, altitude_m, x_station : flight condition + panel location
    emissivity  : surface emissivity (UHTC ceramics: 0.85, polished metal: 0.1)
    tol         : temperature convergence tolerance [K]

    Returns
    -------
    T_wall_eq : float  radiation-equilibrium wall temperature [K]
    """
    sigma = 5.670e-8   # W/(m^2.K^4) Stefan-Boltzmann

    T_lo, T_hi = 200.0, aero_wall_temperature(mach, altitude_m, recovery_factor)

    for _ in range(max_iter):
        T_mid = 0.5 * (T_lo + T_hi)
        q_conv = aero_heat_flux(mach, altitude_m, x_station, T_mid, recovery_factor)
        q_rad  = emissivity * sigma * T_mid**4
        residual = q_conv - q_rad
        if abs(residual) < emissivity * sigma * T_mid**3 * tol:
            return T_mid
        if residual > 0:
            T_lo = T_mid
        else:
            T_hi = T_mid

    return 0.5 * (T_lo + T_hi)


# ---------------------------------------------------------------------------
# Convenience: build ThermalState from flight condition
# ---------------------------------------------------------------------------

def thermal_state_from_flight(
    mach:        float,
    altitude_m:  float,
    x_station:   float,
    T_cure:      float = 450.15,   # 177 C
    emissivity:  float = 0.85,
    gradient_fraction: float = 0.4,
) -> ThermalState:
    """
    Build a ThermalState for a panel at (mach, altitude, x_station).

    The outer-face temperature is the radiation-equilibrium temperature.
    The inner-face temperature is estimated as T_outer * (1 - gradient_fraction),
    approximating the temperature drop across the TPS / skin thickness.
    gradient_fraction = 0 → uniform temperature; 0.4 → 40% cooler on inner face.

    Parameters
    ----------
    mach, altitude_m, x_station : flight condition + chordwise position
    T_cure          : cure temperature [K]  (stress-free reference)
    emissivity      : surface emissivity for radiation equilibrium
    gradient_fraction : fraction of T_outer used for gradient estimate

    Returns
    -------
    ThermalState
    """
    T_outer = equilibrium_wall_temperature(mach, altitude_m, x_station, emissivity)
    T_inner = T_cure + (T_outer - T_cure) * (1.0 - gradient_fraction)

    return ThermalState(
        T_wall_outer = T_outer,
        T_wall_inner = T_inner,
        T_cure       = T_cure,
    )


if __name__ == "__main__":
    import sys as _sys, os as _os
    _sys.stdout.reconfigure(encoding="utf-8")
    _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
    import numpy as np
    from composite_panel.ply import Ply, IM7_8552

    mach, alt = 2.5, 15_000.0   # M=2.5 at 15 km cruise

    T_aw = aero_wall_temperature(mach, alt)
    print(f"M={mach} at {alt/1e3:.0f} km:")
    print(f"  Adiabatic wall temperature  T_aw     = {T_aw:.1f} K  ({T_aw-273.15:.1f} °C)")

    T_eq = equilibrium_wall_temperature(mach, alt, x_station=0.5)
    print(f"  Radiation-equilibrium T_wall          = {T_eq:.1f} K  ({T_eq-273.15:.1f} °C)")

    q = aero_heat_flux(mach, alt, x_station=0.5, T_wall=T_eq)
    print(f"  Heat flux at x=0.5 m (equil. T_wall) = {q:.0f} W/m²")
    print()

    # Thermal load resultants for a [0/45/-45/90]s 8-ply IM7/8552 laminate
    mat        = IM7_8552()
    t_ply      = 0.125e-3
    angles     = [0, 45, -45, 90, 90, -45, 45, 0]
    plies      = [Ply(mat, t_ply, a) for a in angles]
    pt         = IM7_8552_thermal()
    ply_thermals = [pt] * len(plies)

    # z-interfaces for equal-thickness-ply laminate
    h = len(plies) * t_ply
    z_interfaces = np.linspace(-h / 2, h / 2, len(plies) + 1)

    # 40 °C temperature drop across 1 mm skin thickness
    ts = ThermalState(T_wall_outer=T_eq, T_wall_inner=T_eq - 40.0, T_cure=pt.T_cure)
    N_T, M_T = thermal_resultants(plies, ply_thermals, ts, z_interfaces)

    print(f"Thermal resultants for [0/45/-45/90]s at M={mach}, alt={alt/1e3:.0f}km:")
    print(f"  N_T [N/m]    : Nx={N_T[0]:.0f},  Ny={N_T[1]:.0f},  Nxy={N_T[2]:.0f}")
    print(f"  M_T [N·m/m]  : Mx={M_T[0]:.3f},  My={M_T[1]:.3f},  Mxy={M_T[2]:.3f}")

    # Laminate-level CTE
    from composite_panel.laminate import Laminate
    lam   = Laminate(plies)
    alpha = lam.alpha_lam(ply_thermals)
    print(f"  Laminate CTE : αx={alpha[0]*1e6:.2f} µ/K,  αy={alpha[1]*1e6:.2f} µ/K,  αxy={alpha[2]*1e6:.2f} µ/K")
