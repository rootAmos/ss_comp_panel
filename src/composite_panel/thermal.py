"""
Aerodynamic heating and CLT thermal load resultants.

Eckert reference-temperature method for heat flux, cure-residual stresses
from CTE mismatch, through-thickness gradient support.

Ref: Eckert (1955), Kassapoglou (2013) Ch. 7, Jones (1999) Ch. 5
"""

from __future__ import annotations

import aerosandbox.numpy as np
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Union

from composite_panel.ply import PlyMaterial, Ply


@dataclass
class PlyThermal:
    """CTE data for a UD ply: alpha_1 (fibre), alpha_2 (transverse) [1/K]."""
    alpha_1: float
    alpha_2: float
    T_cure:  float = 450.15   # 177 C


def IM7_8552_thermal() -> PlyThermal:
    """IM7/8552 CTEs.  Bowles & Tompkins (1989), NASA RP-1205."""
    return PlyThermal(alpha_1=0.3e-6, alpha_2=28.8e-6, T_cure=450.15)


def alpha_bar(pt: PlyThermal, angle_rad: float) -> np.ndarray:
    """CTE transformed to laminate axes.  Jones (1999), Eq. 2.107."""
    a1, a2 = pt.alpha_1, pt.alpha_2
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array([
        a1 * c**2 + a2 * s**2,
        a1 * s**2 + a2 * c**2,
        2.0 * (a1 - a2) * c * s,
    ])


@dataclass
class ThermalState:
    """Temperature state: outer/inner face temps and cure reference."""
    T_wall_outer: float
    T_wall_inner: float
    T_cure:       float

    @property
    def delta_T_mean(self) -> float:
        return 0.5 * (self.T_wall_outer + self.T_wall_inner) - self.T_cure

    @property
    def delta_T_gradient(self) -> float:
        return self.T_wall_outer - self.T_wall_inner

    def delta_T_at_z(self, z: float, h: float) -> float:
        """Linear interpolation of temperature rise at z in [-h/2, +h/2]."""
        T_z = self.T_wall_inner + (z / h + 0.5) * self.delta_T_gradient
        return T_z - self.T_cure


def thermal_resultants(
    plies: List[Ply], ply_thermals: List[PlyThermal],
    thermal_state: ThermalState, z_interfaces: np.ndarray,
) -> tuple:
    """CLT thermal resultants N_T, M_T.  Jones (1999), Eqs. 4.28-4.29."""
    if len(ply_thermals) != len(plies):
        raise ValueError(
            f"ply_thermals length ({len(ply_thermals)}) != plies ({len(plies)})"
        )

    h_total = z_interfaces[-1] - z_interfaces[0]
    N_T = np.zeros(3)
    M_T = np.zeros(3)

    for k, (ply, pt) in enumerate(zip(plies, ply_thermals)):
        z0    = z_interfaces[k]
        z1    = z_interfaces[k + 1]
        z_mid = 0.5 * (z0 + z1)
        t_k   = z1 - z0

        delta_T_k = thermal_state.delta_T_at_z(z_mid, h_total)
        ab = alpha_bar(pt, np.radians(ply.angle_deg))
        Qb_ab = ply.Q_bar @ ab

        N_T += Qb_ab * delta_T_k * t_k
        M_T += Qb_ab * delta_T_k * z_mid * t_k

    return N_T, M_T


# ---------------------------------------------------------------------------
# Aerodynamic heating
# ---------------------------------------------------------------------------

def aero_wall_temperature(mach: float, altitude_m: float,
                          recovery_factor: float = 0.89) -> float:
    """Adiabatic wall temperature [K].  Anderson (2006), Eq. 6.30.  r=Pr^(1/3)~0.89 for turbulent air."""
    import aerosandbox as asb
    atm   = asb.Atmosphere(altitude=altitude_m)
    T_inf = atm.temperature()
    T_aw = T_inf * (1.0 + recovery_factor * 0.2 * mach**2)
    return float(T_aw)


def aero_heat_flux(mach: float, altitude_m: float, x_station: float,
                   T_wall: float, recovery_factor: float = 0.89) -> float:
    """Convective heat flux [W/m^2].  Eckert (1955) reference-temperature method, turbulent flat plate."""
    import aerosandbox as asb

    atm   = asb.Atmosphere(altitude=altitude_m)
    T_inf = float(atm.temperature())
    P_inf = float(atm.pressure())
    V_inf = mach * float(atm.speed_of_sound())
    R_air = 287.05
    Pr    = 0.71

    T_aw  = aero_wall_temperature(mach, altitude_m, recovery_factor)
    T_star = 0.5 * (T_inf + T_wall) + 0.22 * (T_aw - T_inf)   # Eckert reference temperature

    rho_star = P_inf / (R_air * T_star)
    mu_star  = 1.458e-6 * T_star**1.5 / (T_star + 110.4)   # Sutherland's law for air
    cp_star  = 1.4 * R_air / 0.4                             # cp = gamma*R/(gamma-1)

    Re_x_star = rho_star * V_inf * x_station / mu_star
    if Re_x_star < 1.0:
        return 0.0

    # Chilton-Colburn analogy: St = (Cf/2) * Pr^(-2/3), Cf = 0.0592*Re^(-0.2)
    St_star = 0.0296 * Re_x_star**(-0.2) * Pr**(-2.0 / 3.0)
    return float(St_star * rho_star * V_inf * cp_star * (T_aw - T_wall))


def equilibrium_wall_temperature(mach: float, altitude_m: float, x_station: float,
                                  emissivity: float = 0.85,
                                  recovery_factor: float = 0.89,
                                  tol: float = 1.0, max_iter: int = 50) -> float:
    """Radiation-equilibrium wall temperature [K] via bisection."""
    sigma = 5.670e-8

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


def thermal_state_from_flight(mach: float, altitude_m: float, x_station: float,
                               T_cure: float = 450.15, emissivity: float = 0.85,
                               gradient_fraction: float = 0.4) -> ThermalState:
    """Build ThermalState from flight condition using radiation-equilibrium T_outer."""
    T_outer = equilibrium_wall_temperature(mach, altitude_m, x_station, emissivity)
    T_inner = T_cure + (T_outer - T_cure) * (1.0 - gradient_fraction)
    return ThermalState(T_wall_outer=T_outer, T_wall_inner=T_inner, T_cure=T_cure)


if __name__ == "__main__":
    import sys as _sys, os as _os
    _sys.stdout.reconfigure(encoding="utf-8")
    _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
    from composite_panel.ply import Ply, IM7_8552

    mach, alt = 2.5, 15_000.0

    T_aw = aero_wall_temperature(mach, alt)
    print(f"M={mach} at {alt/1e3:.0f} km:")
    print(f"  T_aw = {T_aw:.1f} K  ({T_aw-273.15:.1f} degC)")

    T_eq = equilibrium_wall_temperature(mach, alt, x_station=0.5)
    print(f"  T_eq = {T_eq:.1f} K  ({T_eq-273.15:.1f} degC)")

    q = aero_heat_flux(mach, alt, x_station=0.5, T_wall=T_eq)
    print(f"  q_dot at equil = {q:.0f} W/m^2")
    print()

    mat = IM7_8552()
    t_ply = 0.125e-3
    angles = [0, 45, -45, 90, 90, -45, 45, 0]
    plies = [Ply(mat, t_ply, a) for a in angles]
    pt = IM7_8552_thermal()
    ply_thermals = [pt] * len(plies)

    h = len(plies) * t_ply
    z_interfaces = np.linspace(-h / 2, h / 2, len(plies) + 1)

    ts = ThermalState(T_wall_outer=T_eq, T_wall_inner=T_eq - 40.0, T_cure=pt.T_cure)
    N_T, M_T = thermal_resultants(plies, ply_thermals, ts, z_interfaces)

    print(f"Thermal resultants for [0/45/-45/90]s:")
    print(f"  N_T: Nx={N_T[0]:.0f},  Ny={N_T[1]:.0f},  Nxy={N_T[2]:.0f} N/m")
    print(f"  M_T: Mx={M_T[0]:.3f},  My={M_T[1]:.3f},  Mxy={M_T[2]:.3f} N*m/m")

    from composite_panel.laminate import Laminate
    lam = Laminate(plies)
    alpha = lam.alpha_lam(ply_thermals)
    print(f"  Laminate CTE: alphax={alpha[0]*1e6:.2f}, alphay={alpha[1]*1e6:.2f} u/K")
