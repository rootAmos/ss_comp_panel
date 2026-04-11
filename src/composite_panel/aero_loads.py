"""
Aerodynamic load generation for panel-level structural sizing.

Pressure models:
  - Karman-Tsien subsonic (M <= 0.85, capped at Cp_crit)
  - Ackeret linearised supersonic (M >= 1.15)
  - Modified Newtonian (hypersonic, standalone)
  - Transonic (0.85 < M < 1.15): not supported -- raises ValueError

All functions are CasADi-compatible: aerosandbox.numpy ops throughout,
no Python conditionals on symbolic values, analytical closed-form integrals.

Ref: Ackeret (1925), Anderson (2003), Kassapoglou (2013) Ch. 1
"""

from __future__ import annotations
import aerosandbox.numpy as np
from dataclasses import dataclass


@dataclass
class PanelLoads:
    """Running loads [N/m] and moments [N*m/m] on a flat panel.
    Compression negative, tension positive."""
    Nxx: float = 0.0
    Nyy: float = 0.0
    Nxy: float = 0.0
    Mxx: float = 0.0   # out-of-plane bending from pressure loading
    Myy: float = 0.0
    Mxy: float = 0.0

    @property
    def N(self) -> np.ndarray:
        """In-plane running load vector [Nxx, Nyy, Nxy] [N/m]."""
        return np.array([self.Nxx, self.Nyy, self.Nxy])

    @property
    def M(self) -> np.ndarray:
        """Running moment vector [Mxx, Myy, Mxy] [N*m/m]."""
        return np.array([self.Mxx, self.Myy, self.Mxy])

    def __str__(self) -> str:
        return (f"PanelLoads  Nxx={self.Nxx/1e3:+.1f} kN/m  "
                f"Nyy={self.Nyy/1e3:+.1f} kN/m  "
                f"Nxy={self.Nxy/1e3:+.1f} kN/m")


# ---------------------------------------------------------------------------
# Wing geometry + multi-panel loads
# ---------------------------------------------------------------------------

@dataclass
class WingGeometry:
    """Trapezoidal wing planform.  eta = y/semi_span (0=root, 1=tip)."""
    semi_span:    float
    root_chord:   float
    taper_ratio:  float
    sweep_le_deg: float
    t_over_c:     float  = 0.04
    mtow_n:       float  = 150_000.0   # N
    ea_offset:    float  = 0.0         # AC-to-EA offset [fraction of chord]
    stringer_pitch_m: float | None = None  # chordwise panel width [m]

    def chord(self, eta):
        """Linear taper: c(eta) = c_root * [1 - (1-lambda)*eta]."""
        return self.root_chord * (1.0 - (1.0 - self.taper_ratio) * eta)

    def box_height(self, eta):
        """Structural box height [m] = t/c * local chord."""
        return self.t_over_c * self.chord(eta)

    def sweep_factor(self):
        """cos^2(sweep_LE)  --  resolves bending load along panel axes."""
        return np.cos(np.radians(self.sweep_le_deg)) ** 2


def wing_panel_loads(
    wing: "WingGeometry",
    eta,
    mach,
    altitude_m,
    alpha_deg,
    n_load = 2.5,
) -> "PanelLoads":
    """Running loads at spanwise station eta for upper wing skin.
    Strip-theory (chord-proportional) spanwise distribution."""
    c_loc = wing.chord(eta)
    b     = wing.semi_span
    y_loc = eta * b

    rho, a_sound = _isa(altitude_m)
    q = 0.5 * rho * (mach * a_sound) ** 2

    # Nyy  --  panel pressure (KT subsonic, Ackeret supersonic)
    delta_p = panel_pressure(mach, alpha_deg, q) * n_load
    b_panel = wing.stringer_pitch_m if wing.stringer_pitch_m is not None else c_loc
    Nyy = -delta_p * b_panel / 2.0

    # Nxx  --  skin as wing-box compression flange, loaded by spanwise bending moment
    L_total = wing.mtow_n * n_load
    M_bend  = _chord_weighted_bending_moment(wing, L_total, y_loc)
    Nxx     = -M_bend / (wing.box_height(eta) * c_loc) * wing.sweep_factor()

    # Nxy  --  sweep transformation of spar compression + Bredt from pitching moment
    #
    # (1) Sweep shear: the spar carries N_spar = Nxx / cos^2(Lambda).
    #     Panel shear from coordinate transform: Nxy = N_spar * sin(L)*cos(L) = Nxx * tan(L).
    #     Ref: Megson "Aircraft Structural Analysis" (2014), Ch. 15.
    sweep_rad = np.radians(wing.sweep_le_deg)
    Nxy = np.abs(Nxx) * np.tan(sweep_rad)

    # (2) Bredt shear flow from pitching moment (ea_offset != 0 activates this).
    #     Pitching moment per unit span: m(xi) = delta_p * ea_offset * c(xi)^2   [N]
    #     Cumulative torque from tip: T(y) = integral_y^b m(xi) dxi  [N*m]
    #     Analytical: c(xi) = c_root*(1 - r*xi), r = (1-lam)/b
    #     T(y) = delta_p * ea_offset * c_root^2 * [(b-y) - r*(b^2-y^2) + r^2*(b^3-y^3)/3]
    r   = (1.0 - wing.taper_ratio) / b
    K_t = delta_p * wing.ea_offset * wing.root_chord ** 2
    T_y = K_t * (
        (b - y_loc)
        - r * (b ** 2 - y_loc ** 2)
        + r ** 2 * (b ** 3 - y_loc ** 3) / 3.0
    )
    A_box = wing.t_over_c * 0.5 * c_loc ** 2        # box cell area [m^2]
    Nxy  += np.abs(T_y) / np.fmax(2.0 * A_box, 1e-12)

    # Mxx  --  local pressure bending (simply-supported panel between stringers)
    Mxx = delta_p * b_panel ** 2 / 8.0

    return PanelLoads(Nxx=Nxx, Nyy=Nyy, Nxy=Nxy, Mxx=Mxx)


# ---------------------------------------------------------------------------
# 1.  Pressure models
# ---------------------------------------------------------------------------

def ackeret_panel_pressure(mach, alpha_deg, q_inf):
    """Ackeret linearised supersonic DeltaCp.  Ackeret (1925); Anderson (2003), Eq. 9.36."""
    beta     = np.sqrt(np.fmax(mach ** 2 - 1.0, 1e-6))
    alpha    = np.radians(alpha_deg)
    delta_Cp = 4.0 * alpha / beta
    return delta_Cp * q_inf


def _cp_max_modified_newtonian(mach, gamma: float = 1.4):
    """Rayleigh Pitot stagnation Cp.  Anderson (2003), Eq. 8.80."""
    g   = gamma
    M2  = mach ** 2
    p02 = (
        ((g + 1.0) ** 2 * M2 / (4.0 * g * M2 - 2.0 * (g - 1.0))) ** (g / (g - 1.0))
        * (2.0 * g * M2 - (g - 1.0)) / (g + 1.0)
    )
    return (p02 - 1.0) / (0.5 * g * M2)


def hypersonic_panel_pressure(mach, alpha_deg, q_inf, gamma: float = 1.4):
    """Modified Newtonian pressure.  Lees (1955); Anderson (2006), Eq. 3.15."""
    alpha    = np.radians(alpha_deg)
    Cp_max   = _cp_max_modified_newtonian(mach, gamma)
    delta_Cp = Cp_max * np.sin(alpha) ** 2
    return delta_Cp * q_inf


_M_SUB_MAX = 0.85   # Karman-Tsien valid below this
_M_SUP_MIN = 1.15   # Ackeret valid above this


def _cp_crit(mach, gamma: float = 1.4):
    """Critical Cp (local sonic condition).  Anderson (2003), Eq. 11.38."""
    g    = gamma
    M2   = mach ** 2
    base = (2.0 / (g + 1.0)) * (1.0 + (g - 1.0) / 2.0 * M2)
    return (2.0 / (g * M2)) * (base ** (g / (g - 1.0)) - 1.0)


def _kt_delta_cp(alpha_rad, mach, gamma: float = 1.4):
    """Karman-Tsien compressibility correction for flat-plate DeltaCp, capped at Cp_crit.
    Anderson (2003), Eq. 11.54; von Karman (1941), J. Aeronautical Sciences."""
    Cp0   = 2.0 * np.pi * alpha_rad
    beta  = np.sqrt(np.fmax(1.0 - mach ** 2, 1e-9))
    denom = np.fmax(beta + mach ** 2 * Cp0 / (2.0 * (1.0 + beta)), 1e-12)
    Cp_kt = Cp0 / denom
    Cp_cap = -2.0 * _cp_crit(mach, gamma)   # positive upper bound
    return np.fmin(Cp_kt, Cp_cap)


def panel_pressure(mach, alpha_deg, q_inf, gamma: float = 1.4):
    """
    Panel pressure [Pa].  KT subsonic (M <= 0.85), Ackeret supersonic (M >= 1.15).
    Raises ValueError in the transonic gap -- no valid closed-form model exists there.
    """
    if _M_SUB_MAX < mach < _M_SUP_MIN:
        raise ValueError(
            f"Mach {mach:.3f} is in the transonic regime "
            f"({_M_SUB_MAX} < M < {_M_SUP_MIN}).  "
            f"No valid closed-form pressure model implemented for this regime in this model."
        )
    if mach <= _M_SUB_MAX:
        alpha = np.radians(alpha_deg)
        return _kt_delta_cp(alpha, mach, gamma) * q_inf
    else:
        return ackeret_panel_pressure(mach, alpha_deg, q_inf)


# ---------------------------------------------------------------------------
# 2.  Spanwise bending moment (strip theory)
# ---------------------------------------------------------------------------

def _chord_weighted_bending_moment(wing: "WingGeometry", L_total, y):
    """Bending moment at station y for chord-proportional (strip-theory) load.

    Strip theory assumes l(y) proportional to c(y) — valid when Mach cones
    confine disturbances to the local strip (linearised supersonic theory).

    M(y) = integral_y^b l(xi)*(xi-y) dxi,  l(xi) = K*(1 - r*xi).
    Evaluated analytically:
        integral (1-r*xi)(xi-y) dxi  =  (b-y)^2/2
                                       - r*[(b^3-y^3)/3 - y*(b^2-y^2)/2]
    """
    b   = wing.semi_span
    lam = wing.taper_ratio
    # Amplitude: A = (L/2) / S_half,  K = A * c_root
    S_half = wing.root_chord * b * (1.0 + lam) / 2.0
    K   = (L_total / 2.0) / S_half * wing.root_chord
    r   = (1.0 - lam) / b
    # Derived from integral_y^b A*c(xi)*(xi-y) dxi, c(xi) = c_root*(1 - r*xi)
    return K * (
        (b - y) ** 2 / 2.0
        - r * ((b ** 3 - y ** 3) / 3.0 - y * (b ** 2 - y ** 2) / 2.0)
    )


def spanwise_lift_distribution(
    wing: "WingGeometry",
    L_total: float,
    n: int = 200,
) -> "tuple[np.ndarray, np.ndarray]":
    """Chord-proportional spanwise l(y) [N/m] for visualisation.  Returns (y, l) arrays.
    Strip theory: l(y) = (L/2) * c(y) / S_half.  Same linearised supersonic
    assumption as _chord_weighted_bending_moment."""
    b   = wing.semi_span
    lam = wing.taper_ratio

    y_arr = np.linspace(0.0, b, n)

    S_half  = wing.root_chord * b * (1.0 + lam) / 2.0
    A       = (L_total / 2.0) / S_half
    c_y     = wing.root_chord * (1.0 - (1.0 - lam) * y_arr / b)
    l_arr   = A * c_y

    return y_arr, l_arr


# ---------------------------------------------------------------------------
# ISA atmosphere model
# ---------------------------------------------------------------------------

def _isa(altitude_m):
    """ISA atmosphere (troposphere + stratosphere).  Returns (rho, a_sound).
    ICAO Doc 7488/3 (1993); ISO 2533:1975."""
    T0    = 288.15    # K    sea-level temperature
    P0    = 101325.0  # Pa   sea-level pressure
    L     = -0.0065   # K/m  troposphere lapse rate
    g     = 9.80665   # m/s^2
    R     = 287.05    # J/(kg*K)
    gamma = 1.4

    # Troposphere (0 – 11 km): linear T, power-law P
    T_trop = T0 + L * altitude_m
    P_trop = P0 * (T_trop / T0) ** (-g / (L * R))

    # Stratosphere (>11 km): isothermal, exponential P
    T11    = T0 + L * 11000.0
    P11    = P0 * (T11 / T0) ** (-g / (L * R))
    T_strat = T11
    P_strat = P11 * np.exp(-g * (altitude_m - 11000.0) / (R * T11))

    # CasADi-compatible branch selection
    T = np.where(altitude_m <= 11000.0, T_trop, T_strat)
    P = np.where(altitude_m <= 11000.0, P_trop, P_strat)

    rho = P / (R * T)
    a   = np.sqrt(gamma * R * T)
    return rho, a


if __name__ == "__main__":
    import sys as _sys
    _sys.stdout.reconfigure(encoding="utf-8")

    wing = WingGeometry(
        semi_span    = 4.5,
        root_chord   = 2.0,
        taper_ratio  = 0.3,
        sweep_le_deg = 45.0,
        t_over_c     = 0.04,
        mtow_n       = 150_000.0,
    )

    # Wing panel loads at three spanwise stations
    print("Wing panel loads  (M=1.7, alt=15 km, alpha=3.5deg, 2.5g):")
    print(f"  {'eta':>5}  {'Nxx (kN/m)':>12}  {'Nyy (kN/m)':>12}  {'Nxy (kN/m)':>12}  {'Mxx (N*m/m)':>13}")
    for eta in [0.1, 0.4, 0.7]:
        wl = wing_panel_loads(wing, eta, mach=1.7, altitude_m=15_000.0,
                              alpha_deg=3.5, n_load=2.5)
        print(f"  {eta:5.2f}  {wl.Nxx/1e3:12.1f}  {wl.Nyy/1e3:12.1f}  "
              f"{wl.Nxy/1e3:12.1f}  {wl.Mxx:13.1f}")
    print()

    # Panel pressure at subsonic and supersonic points
    print("Panel pressure  (alt=15 km, alpha=5deg):")
    print(f"  {'M':>5}  {'Deltap (kPa)':>12}")
    for M in [0.3, 0.5, 0.8, 1.2, 1.5, 1.7, 2.5, 5.0]:
        rho_m, a_m = _isa(15_000.0)
        q_m = 0.5 * rho_m * (a_m * M) ** 2
        dp  = panel_pressure(M, 5.0, q_m)
        print(f"  {M:5.1f}  {dp/1e3:12.3f}")
