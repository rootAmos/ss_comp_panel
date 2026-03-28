"""
composite_panel.aero_loads
--------------------------
Simplified aerodynamic load generation for panel-level structural analysis.

Two approaches:
  1. Supersonic linear theory (Ackeret) – fast, closed-form, suitable for
     conceptual sizing of a Mach 1.4–3.0 wing skin panel.
  2. Elliptic spanwise lift distribution → running loads on a given panel.

These are intentionally simple – the point is to generate physically
meaningful load vectors [Nxx, Nyy, Nxy] to feed into CLT, not to replace
a full aero solver.

Reference:
    Anderson, J.D. – Modern Compressible Flow (McGraw-Hill, 2003), Ch. 9
    Kassapoglou, C. – Design and Analysis of Composite Structures, Ch. 1
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass
class PanelLoads:
    """
    Running loads on a flat panel [N/m].

    Sign convention: tensile Nxx positive.
    """
    Nxx: float = 0.0   # spanwise (chord-parallel)
    Nyy: float = 0.0   # chordwise
    Nxy: float = 0.0   # in-plane shear
    Mxx: float = 0.0   # spanwise bending moment [N·m/m]
    Myy: float = 0.0   # chordwise bending moment
    Mxy: float = 0.0   # twisting moment

    @property
    def N(self) -> np.ndarray:
        return np.array([self.Nxx, self.Nyy, self.Nxy])

    @property
    def M(self) -> np.ndarray:
        return np.array([self.Mxx, self.Myy, self.Mxy])

    def __str__(self) -> str:
        return (f"PanelLoads  Nxx={self.Nxx/1e3:+.1f} kN/m  "
                f"Nyy={self.Nyy/1e3:+.1f} kN/m  "
                f"Nxy={self.Nxy/1e3:+.1f} kN/m")


# ---------------------------------------------------------------------------
# 1.  Ackeret (linearised supersonic) pressure on a flat plate
# ---------------------------------------------------------------------------

def ackeret_panel_pressure(mach: float,
                            alpha_deg: float,
                            q_inf: float) -> float:
    """
    Ackeret 2-D linearised pressure coefficient for a flat plate.

    ΔCp = 4α / sqrt(M²-1)    →   Δp = ΔCp × q∞

    Parameters
    ----------
    mach      : free-stream Mach number (> 1)
    alpha_deg : angle of attack [degrees]
    q_inf     : dynamic pressure [Pa]  = 0.5 * ρ * V²

    Returns
    -------
    Δp : net pressure difference (lower - upper) [Pa]
    """
    if mach <= 1.0:
        raise ValueError(f"Ackeret theory requires Mach > 1 (got {mach})")
    beta = np.sqrt(mach**2 - 1.0)
    alpha = np.radians(alpha_deg)
    delta_Cp = 4.0 * alpha / beta
    return delta_Cp * q_inf


def supersonic_panel_loads(mach: float,
                            altitude_m: float,
                            alpha_deg: float,
                            panel_chord: float,
                            panel_span:  float,
                            skin_thickness_estimate: float = 2e-3,
                            n_load: float = 2.5) -> PanelLoads:
    """
    Estimate running loads on an upper-surface wing skin panel
    for a supersonic cruise condition.

    Parameters
    ----------
    mach                    : cruise Mach number
    altitude_m              : cruise altitude [m]
    alpha_deg               : angle of attack [deg]
    panel_chord             : panel dimension in chordwise direction [m]
    panel_span              : panel dimension in spanwise direction [m]
    skin_thickness_estimate : panel skin thickness (for bending estimate) [m]
    n_load                  : ultimate load factor (design)

    Returns
    -------
    PanelLoads
    """
    # ISA atmosphere (simple model)
    rho, a = _isa(altitude_m)
    V     = mach * a
    q_inf = 0.5 * rho * V**2

    # Net pressure (Ackeret)
    delta_p = ackeret_panel_pressure(mach, alpha_deg, q_inf) * n_load

    # Chordwise running load from pressure × chord → Nyy (dominant compression on upper skin)
    Nyy = -delta_p * panel_chord / 2.0        # compression on upper skin (half-span assumption)

    # Spanwise bending induces Nxx: rough beam-bending estimate
    # Treat panel as flange of a wing box – very simplified
    Nxx = Nyy * 0.6    # typical Nxx/Nyy ratio for swept supersonic planform

    # Torsion → shear flow in skin
    Nxy = abs(Nxx) * 0.25

    # Out-of-plane bending moment from pressure
    Mxx = delta_p * panel_chord**2 / 8.0

    return PanelLoads(Nxx=Nxx, Nyy=Nyy, Nxy=Nxy, Mxx=Mxx)


# ---------------------------------------------------------------------------
# 2.  Elliptic lift distribution → panel running load
# ---------------------------------------------------------------------------

def elliptic_spanwise_load(MTOW_N: float,
                            n_load: float,
                            semi_span: float,
                            eta: float,
                            chord_at_eta: float) -> PanelLoads:
    """
    Elliptic spanwise lift distribution → spanwise running load at station η.

    Parameters
    ----------
    MTOW_N       : max take-off weight [N]
    n_load       : load factor
    semi_span    : wing semi-span [m]
    eta          : dimensionless spanwise station  (0 = root, 1 = tip)
    chord_at_eta : local chord at η [m]

    Returns
    -------
    PanelLoads  (Nyy dominant – chordwise compression from bending)
    """
    L_total   = MTOW_N * n_load
    # Elliptic distribution: l(y) = l0 * sqrt(1 - (y/b)²)
    # l0 = 4L / (π * 2b)  where 2b = full span
    b   = semi_span
    l0  = 4.0 * L_total / (np.pi * 2.0 * b)
    y   = eta * b

    l_y = l0 * np.sqrt(max(1.0 - (y/b)**2, 0.0))   # N/m (spanwise running load)

    # Bending moment at η (outboard integral)
    # M(η) ≈ ∫_y^b l(ξ)(ξ-y)dξ  –  closed form for elliptic:
    M_bend = _elliptic_bending_moment(l0, b, y)

    # Running load in panel skin (upper surface compression)
    # Very rough: Nxx ≈ M_bend / (z_arm × chord_at_eta)
    z_arm  = 0.12 * chord_at_eta    # ~12% chord wing-box height assumption
    Nxx    = -M_bend / (z_arm * chord_at_eta)  # compression on upper skin

    return PanelLoads(Nxx=Nxx, Nyy=Nxx * 0.3, Nxy=abs(Nxx) * 0.15)


def _elliptic_bending_moment(l0: float, b: float, y: float) -> float:
    """Bending moment at spanwise station y for elliptic load distribution."""
    # Analytical result: M(y) = l0 * [ b/2 * sqrt(b²-y²) - y²/2 * arccos(y/b)
    #                                   + (b²-y²)^(3/2)/(3b) ]  ... simplified
    # Using numerical integration for reliability
    n  = 500
    xi = np.linspace(y, b, n)
    l  = l0 * np.sqrt(np.maximum(1.0 - (xi/b)**2, 0.0))
    M  = np.trapz(l * (xi - y), xi)
    return M


# ---------------------------------------------------------------------------
# ISA atmosphere
# ---------------------------------------------------------------------------

def _isa(altitude_m: float):
    """
    Simple ISA model up to 20 km.
    Returns (density [kg/m³], speed of sound [m/s]).
    """
    T0, P0, rho0 = 288.15, 101325.0, 1.225   # sea level
    L  = -0.0065   # lapse rate K/m  (troposphere)
    g  = 9.80665
    R  = 287.05
    gamma = 1.4

    if altitude_m <= 11000:
        T   = T0 + L * altitude_m
        P   = P0 * (T / T0) ** (-g / (L * R))
    else:
        # Stratosphere – isothermal
        T11 = T0 + L * 11000
        P11 = P0 * (T11 / T0) ** (-g / (L * R))
        T   = T11
        P   = P11 * np.exp(-g * (altitude_m - 11000) / (R * T11))

    rho = P / (R * T)
    a   = np.sqrt(gamma * R * T)
    return rho, a
