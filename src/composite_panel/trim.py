"""
Trim AoA from lift balance.  Subsonic (Prandtl-Glauert + Helmbold) and
supersonic (Ackeret + Jones) finite-wing CLalpha.  Transonic not supported.

Ref: Anderson (2003), 'Fundamentals of Aerodynamics', 3rd ed.
     Helmbold (1942), Jahrbuch der Luftfahrtforschung
     Jones (1947), NACA TN 1107
"""

from __future__ import annotations

import aerosandbox.numpy as np
from dataclasses import dataclass
from typing import Optional

from composite_panel.aero_loads import WingGeometry, _isa as isa_atmosphere



@dataclass
class TrimState:
    """Trim state: alpha, CL, CLalpha, q_inf at a given flight condition."""
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
                f"n={self.n_load:.2f}g  alpha={self.alpha_deg:.2f}deg  "
                f"CL={self.CL:.4f}  CLalpha={self.CLalpha:.3f}/rad  "
                f"q?={self.q_inf/1e3:.2f}kPa  S={self.S_ref:.2f}m^2")



def oswald_efficiency(aspect_ratio: float, taper_ratio: float,
                      sweep_le_deg: float) -> float:
    """Simplified Oswald efficiency.  Empirical fit — source: simplified form of
    Nita & Scholz (2012), 'Estimating the Oswald Factor from Basic Aircraft
    Geometrical Parameters', DLRK.  Not the full correlation."""
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
    """CLalpha [rad^-1].  PG+Helmbold (subsonic) or Ackeret+Jones (supersonic)."""
    if _M_SUB < mach < _M_SUP:
        raise ValueError(
            f"Mach {mach:.3f} is in the transonic regime ({_M_SUB} < M < {_M_SUP}). "
            f"Neither Prandtl-Glauert nor Ackeret is valid here. "
            f"Use M <= {_M_SUB} (subsonic) or M >= {_M_SUP} (supersonic)."
        )

    if mach <= _M_SUB:
        beta   = np.sqrt(max(1.0 - mach**2, 1e-6))
        CLa_2D = 2.0 * np.pi / beta                          # Prandtl-Glauert; Anderson (2003), Eq. 11.51
        if aspect_ratio is None:
            return CLa_2D
        e = oswald_efficiency(aspect_ratio, taper_ratio, sweep_le_deg)
        return CLa_2D / (1.0 + CLa_2D / (np.pi * aspect_ratio * e))  # Helmbold (1942); Anderson (2003), Eq. 5.69
    else:
        beta   = np.sqrt(max(mach**2 - 1.0, 1e-6))
        CLa_2D = 4.0 / beta                                    # Ackeret (1925); Anderson (2003), Eq. 9.36
        if aspect_ratio is None:
            return CLa_2D
        # Jones (1947), NACA TN 1107: supersonic finite-wing correction.
        # Oswald e is omitted because at supersonic speeds each strip is
        # aerodynamically independent (Mach cone decouples spanwise stations),
        # so the induced-drag efficiency factor does not apply.
        return CLa_2D / np.sqrt(1.0 + (CLa_2D / (np.pi * aspect_ratio)) ** 2)  # Jones (1947), NACA TN 1107



def trim_alpha(
    wing:        WingGeometry,
    mach:        float,
    altitude_m:  float,
    n_load:      float = 1.0,
    alpha_max_deg: float = 20.0,
) -> TrimState:
    """Equilibrium trim AoA.  converged=False if |alpha| > alpha_max_deg."""
    # == Atmosphere ============================================================
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
    alpha_deg = np.degrees(alpha_rad)
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
    """Trim states across a range of Mach numbers."""
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
    print("Trim table  --  15 km, 1g cruise:")
    print(f"  {'M':>5}  {'alpha (deg)':>8}  {'CL':>8}  {'CLalpha (1/rad)':>12}  {'q? (kPa)':>10}")
    for s in trim_table(wing, mach_range, altitude_m=15_000.0, n_load=1.0):
        print(f"  {s.mach:5.2f}  {s.alpha_deg:8.3f}  {s.CL:8.4f}  "
              f"{s.CLalpha:12.4f}  {s.q_inf/1e3:10.2f}")
