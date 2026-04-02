"""
composite_panel.trim
--------------------
Equilibrium trim AoA from lift balance:

    CL_req = n_load * W / (q_inf * S_ref)
    alpha  = CL_req / CLalpha

CLalpha dispatch:
    M <= 0.85 : Prandtl-Glauert 2D + Helmbold finite-wing (AR, Oswald e)
    M >= 1.15 : Ackeret 2D + Jones (1947) supersonic finite-wing
    0.85-1.15 : linear blend

Refs: Helmbold (1942), Jones NACA TN 1107 (1947), Nita & Scholz DLRK 2012
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

try:
    from .aero_loads import WingGeometry, _isa as isa_atmosphere
except ImportError:
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
    from composite_panel.aero_loads import WingGeometry, _isa as isa_atmosphere



@dataclass
class TrimState:
    """
    Aerodynamic trim state for a given flight condition.

    Attributes
    ----------
    mach         : flight Mach number
    altitude_m   : pressure altitude [m]
    n_load       : normal load factor (1.0 = level, 2.5 = maneuver)
    alpha_deg    : trim angle of attack [degrees]
    CL           : trim lift coefficient
    CLalpha      : lift curve slope [rad⁻¹]
    q_inf        : dynamic pressure [Pa]
    S_ref        : reference wing area [m²]
    converged    : True if lift balance converged
    """
    mach:       float
    altitude_m: float
    n_load:     float
    alpha_deg:  float
    CL:         float
    CLalpha:    float
    q_inf:      float
    S_ref:      float
    converged:  bool = True

    def __str__(self) -> str:
        return (f"TrimState  M={self.mach:.2f}  alt={self.altitude_m/1e3:.0f}km  "
                f"n={self.n_load:.2f}g  α={self.alpha_deg:.2f}°  "
                f"CL={self.CL:.4f}  CLα={self.CLalpha:.3f}/rad  "
                f"q∞={self.q_inf/1e3:.2f}kPa  S={self.S_ref:.2f}m²")



def oswald_efficiency(aspect_ratio: float, taper_ratio: float,
                      sweep_le_deg: float) -> float:
    """
    Oswald span efficiency factor e, used to correct the finite-wing CLα.

    Uses the Nita-Scholz (2012) correlation:
        e = 1 / (1 + 0.85 · (1 - λ)² · AR + 4.61 · (1 - 0.045·AR^0.68) · cos(Λ)^0.15 - 1)

    Simplified form used here (conservative for swept tapered wings):
        e ≈ 1 / (1 + δ)
        δ = 0.04 + 0.85 · (1 - taper)²          (drag due to non-elliptic lift)
            + 0.01 · sweep_le_deg / 30           (sweep contribution)

    Returns
    -------
    e : float  [0 < e ≤ 1]
    """
    delta = (0.04
             + 0.85 * (1.0 - taper_ratio) ** 2
             + 0.01 * sweep_le_deg / 30.0)
    return 1.0 / (1.0 + delta)



_M_SUB = 0.85
_M_SUP = 1.15


def lift_curve_slope(mach: float,
                     aspect_ratio: Optional[float] = None,
                     taper_ratio: float = 0.3,
                     sweep_le_deg: float = 0.0) -> float:
    """
    Finite-wing lift curve slope  CLα  [rad⁻¹].

    For 2-D (infinite wing), set aspect_ratio=None.

    Regime dispatch:
      M ≤ 0.85  : Prandtl-Glauert (2D) + Helmbold finite-wing correction
      M ≥ 1.15  : Ackeret (2D) + Jones supersonic finite-wing correction
      0.85–1.15 : linear blend (avoids singularity at M=1)

    Parameters
    ----------
    mach         : Mach number
    aspect_ratio : geometric AR = b²/S_ref  (None → return 2D slope)
    taper_ratio  : tip chord / root chord  (used for Oswald efficiency)
    sweep_le_deg : leading-edge sweep [deg]  (used for Oswald efficiency)

    Returns
    -------
    CLα : float  [rad⁻¹]
    """
    def _sub(M):
        beta   = math.sqrt(max(1.0 - M**2, 1e-6))
        CLa_2D = 2.0 * math.pi / beta                          # Prandtl-Glauert
        if aspect_ratio is None:
            return CLa_2D
        e = oswald_efficiency(aspect_ratio, taper_ratio, sweep_le_deg)
        return CLa_2D / (1.0 + CLa_2D / (math.pi * aspect_ratio * e))  # Helmbold

    def _sup(M):
        beta   = math.sqrt(max(M**2 - 1.0, 1e-6))
        CLa_2D = 4.0 / beta                                    # Ackeret
        if aspect_ratio is None:
            return CLa_2D
        return CLa_2D / math.sqrt(1.0 + (CLa_2D / (math.pi * aspect_ratio)) ** 2)  # Jones

    if mach <= _M_SUB:
        return _sub(mach)
    elif mach >= _M_SUP:
        return _sup(mach)
    else:
        t = (mach - _M_SUB) / (_M_SUP - _M_SUB)
        return (1.0 - t) * _sub(_M_SUB) + t * _sup(_M_SUP)



def trim_alpha(
    wing:        WingGeometry,
    mach:        float,
    altitude_m:  float,
    n_load:      float = 1.0,
    alpha_max_deg: float = 20.0,
) -> TrimState:
    """
    Equilibrium trim AoA.  converged=False if |alpha| > alpha_max_deg.
    Linear aero — preliminary sizing only, validate against CFD for final.
    """
    # ── Atmosphere ────────────────────────────────────────────────────────────
    rho, a_sound = isa_atmosphere(altitude_m)
    V     = mach * a_sound
    q_inf = 0.5 * rho * V ** 2

    tip_chord = wing.root_chord * wing.taper_ratio
    S_ref     = wing.semi_span * (wing.root_chord + tip_chord)   # full trapezoidal area
    AR        = (2 * wing.semi_span) ** 2 / S_ref

    CL_req = n_load * wing.mtow_n / (q_inf * S_ref)

    CLa = lift_curve_slope(mach,
                           aspect_ratio=AR,
                           taper_ratio=wing.taper_ratio,
                           sweep_le_deg=wing.sweep_le_deg)

    alpha_rad = CL_req / max(CLa, 0.1)
    alpha_deg = math.degrees(alpha_rad)
    converged = abs(alpha_deg) < alpha_max_deg

    return TrimState(
        mach       = mach,
        altitude_m = altitude_m,
        n_load     = n_load,
        alpha_deg  = alpha_deg,
        CL         = CL_req,
        CLalpha    = CLa,
        q_inf      = q_inf,
        S_ref      = S_ref,
        converged  = converged,
    )


def trim_table(wing: WingGeometry,
               mach_range,
               altitude_m: float,
               n_load: float = 1.0) -> list:
    """
    Compute trim states across a range of Mach numbers.

    Returns a list of TrimState objects, one per Mach entry.
    Useful for generating loads over a mission profile.

    Example
    -------
    >>> states = trim_table(wing, [0.6, 0.8, 1.2, 1.7, 2.4, 5.0],
    ...                     altitude_m=15000, n_load=2.5)
    >>> for s in states:
    ...     print(s)
    """
    return [trim_alpha(wing, M, altitude_m, n_load) for M in mach_range]


if __name__ == "__main__":
    import sys as _sys, os as _os
    _sys.stdout.reconfigure(encoding="utf-8")
    _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
    from composite_panel.aero_loads import WingGeometry

    wing = WingGeometry(
        semi_span    = 4.5,
        root_chord   = 2.0,
        taper_ratio  = 0.3,
        sweep_le_deg = 45.0,
        t_over_c     = 0.04,
        mtow_n       = 150_000.0,
    )

    # Single-point trim: 1g cruise at M=1.7, 15 km
    state = trim_alpha(wing, mach=1.7, altitude_m=15_000.0, n_load=1.0)
    print("1g cruise trim  (M=1.7, 15 km):")
    print(f"  {state}")
    print()

    # 2.5g manoeuvre trim at the same condition
    state_man = trim_alpha(wing, mach=1.7, altitude_m=15_000.0, n_load=2.5)
    print("2.5g manoeuvre trim  (M=1.7, 15 km):")
    print(f"  {state_man}")
    print()

    # Sweep across mission profile
    mach_range = [0.6, 0.8, 1.2, 1.5, 1.7, 2.0, 2.5]
    print("Trim table — 15 km, 1g cruise:")
    print(f"  {'M':>5}  {'α (°)':>8}  {'CL':>8}  {'CLα (1/rad)':>12}  {'q∞ (kPa)':>10}")
    for s in trim_table(wing, mach_range, altitude_m=15_000.0, n_load=1.0):
        print(f"  {s.mach:5.2f}  {s.alpha_deg:8.3f}  {s.CL:8.4f}  "
              f"{s.CLalpha:12.4f}  {s.q_inf/1e3:10.2f}")
