"""
composite_panel.buckling
------------------------
Composite panel buckling under combined in-plane loads.

BACKGROUND
==========
For thin-skin hypersonic panels, the governing failure mode is frequently
panel buckling rather than material strength (Tsai-Wu).  A 2 mm IM7/8552
skin at 40% semi-span under 6 kN/m compression buckles before reaching
fibre tensile strength — particularly for narrow panels (small a/b ratio)
and high off-axis ply fractions.

FORMULATION
===========
Simply-supported rectangular orthotropic plate (a x b), where:
  a  = spanwise panel dimension [m]   (direction of primary compression Nxx)
  b  = chordwise panel dimension [m]

The exact closed-form solution (Timoshenko & Gere 1961, Ch. 9) gives:

Uniaxial compression  Nxx only:
  N_xcr = (pi^2 / b^2) * min_m [ D11*(m*b/a)^2 + 2*(D12+2*D66) + D22*(a/(m*b))^2 ]

Biaxial compression  Nxx + lambda*Nxx (lambda = Nyy/Nxx):
  Replace Nxx_cr: N_xcr_biax = N_xcr / (1 + lambda * (m*b/a)^2 / (a/m*b)^...)
  (use interaction formula below instead — simpler and conservative)

Shear  Nxy only — no exact closed form for orthotropy.
Approximation (Seydel 1933 modified for orthotropy, ESDU 02.03.11):
  N_xycr = k_s * (pi^2 / b^2) * (D11 * D22^3)^(1/4)
  where k_s = 4 * ( 1 + D12/(D11*D22)^(1/2) + 2*D66/(D11*D22)^(1/2) )^(1/2)
  ... or use the simpler lower bound: k_s = 8.125 for square, 5.35 for long panel

Combined loading — interaction formula (Whitney 1987, ESDU 02.03.11):
  R_x + R_y + R_s^2 = 1     at buckling  (R = load / critical load)
  RF_buckle = positive root of:  R_x*RF + R_y*RF + R_s^2*RF^2 - 1 = 0

  where R_x = |Nxx| / N_xcr,  R_y = |Nyy| / N_ycr,  R_s = |Nxy| / N_xycr
  Note: N_ycr computed analogously with a <-> b, D11 <-> D22.

DIFFERENTIABILITY
=================
For CasADi / IPOPT optimization, N_xcr must be differentiable w.r.t. the
ply thicknesses t_k.  Since D_ij are cubic in t_k (via z^3 integrals),
N_xcr is also cubic in t_k — smooth and well-conditioned for gradient-based
optimization.

The min_m in the uniaxial formula is avoided by fixing the buckling mode
number m* at the geometry ratio a/b (m* = round(a/b) for isotropic plates;
a conservative choice m*=1 is used here as the default).

PANEL DIMENSIONS
================
Typical wing skin panel between adjacent ribs and stringers:
  a (span between ribs)    : 0.4 — 0.8 m
  b (chord between stringers): 0.1 — 0.25 m
  a/b ratio                : 2 — 6  (long panel → m* = 1 or 2)

References
----------
Timoshenko, S.P. & Gere, J.M.  Theory of Elastic Stability (1961) Ch. 9
Whitney, J.M.  Structural Analysis of Laminated Anisotropic Plates (1987)
ESDU 02.03.11  Buckling of Flat Orthotropic Plates
Jones, R.M.    Mechanics of Composite Materials (1999) Ch. 5
"""

from __future__ import annotations

import warnings
import numpy as _np
from typing import Optional, Tuple, Union

import aerosandbox.numpy as np   # CasADi-compatible for smooth buckling constraints


# ---------------------------------------------------------------------------
# D-matrix component extraction and validation
# ---------------------------------------------------------------------------

def _D_components(D: _np.ndarray) -> Tuple:
    """Extract D11, D12, D22, D66 from the (3,3) D matrix."""
    return D[0, 0], D[0, 1], D[1, 1], D[2, 2]


def _check_bend_twist_coupling(D: _np.ndarray, tol: float = 0.05) -> None:
    """
    Warn if D16 or D26 are non-negligible relative to the principal bending stiffness.

    The Timoshenko/Seydel buckling formulas used in this module assume a balanced
    orthotropic plate (D16 = D26 = 0).  For unbalanced laminates — odd-angle plies
    without matching -θ partners, or non-symmetric layups — D16 and D26 are non-zero
    and the closed-form critical loads become unconservative (they overpredict Ncr).

    The check uses:
        coupling_ratio = max(|D16|, |D26|) / sqrt(D11 * D22)

    A ratio above `tol` (default 5%) indicates meaningful bend-twist coupling.
    FEA or Rayleigh-Ritz with full anisotropy should be used in that regime.
    """
    D11, D22 = float(D[0, 0]), float(D[1, 1])
    D16, D26 = float(D[0, 2]), float(D[1, 2])
    ref = _np.sqrt(_np.maximum(D11 * D22, 1e-30))
    ratio = max(abs(D16), abs(D26)) / ref
    if ratio > tol:
        warnings.warn(
            f"Laminate has significant bend-twist coupling: "
            f"max(|D16|,|D26|)/sqrt(D11·D22) = {ratio:.3f} > {tol:.2f}. "
            f"The orthotropic buckling formula (D16=D26=0) will overpredict Ncr. "
            f"Use a balanced layup or validate with FEA.",
            stacklevel=3,
        )


# ---------------------------------------------------------------------------
# Critical loads — numpy (exact, for post-processing)
# ---------------------------------------------------------------------------

def Nxx_cr(
    D:    _np.ndarray,
    a:    float,
    b:    float,
    n_modes: int = 8,
) -> float:
    """
    Critical uniaxial compression load N_xcr [N/m] for simply-supported plate.

    Exact Rayleigh-Ritz result for balanced orthotropic plate (D16=D26=0):
      N_xcr = (pi^2 / b^2) * min_m [ D11*(m*b/a)^2 + 2*(D12+2*D66) + D22*(a/(m*b))^2 ]

    The mode number m giving the minimum load is found by evaluating over
    m = 1 .. n_modes.

    Parameters
    ----------
    D       : (3,3) bending stiffness matrix [N.m]
    a       : panel dimension in loading direction [m]
    b       : panel dimension transverse to loading [m]
    n_modes : number of half-wave modes to evaluate (default 8)

    Returns
    -------
    N_xcr : float  critical compression load [N/m]  (positive = compression)
    """
    D11, D12, D22, D66 = _D_components(D)
    pi2_b2 = _np.pi**2 / b**2
    two_D12_D66 = 2.0 * (D12 + 2.0 * D66)

    m = _np.arange(1, n_modes + 1, dtype=float)
    ratios = m * b / a
    loads = pi2_b2 * (D11 * ratios**2 + two_D12_D66 + D22 / ratios**2)
    return float(_np.min(loads))


def Nyy_cr(D: _np.ndarray, a: float, b: float, n_modes: int = 8) -> float:
    """
    Critical uniaxial compression in y-direction [N/m].
    Swaps (a,b) and (D11,D22) relative to Nxx_cr.
    """
    D_swapped = _np.array([
        [D[1, 1], D[0, 1], D[2, 2]],
        [D[0, 1], D[0, 0], D[2, 2]],
        [D[2, 2], D[2, 2], D[2, 2]],
    ])
    return Nxx_cr(D_swapped, b, a, n_modes)


def Nxy_cr(D: _np.ndarray, a: float, b: float) -> float:
    """
    Critical shear buckling load N_xycr [N/m] for simply-supported plate.

    Approximation from Seydel (1933) / ESDU 02.03.11 for orthotropic plates:
      N_xycr = k_xy * pi^2 / b^2 * (D11 * D22^3)^(1/4)

    with the orthotropic shear buckling coefficient:
      k_xy = 8.125 + 5.045/eta   (for a/b >= 1, conservative for long panels)
      eta  = (D12 + 2*D66) / (D11*D22)^(1/2)   orthotropy parameter

    For a/b < 1, use b < a convention (swap).

    Returns
    -------
    N_xycr : float  critical shear load [N/m]  (positive value)
    """
    D11, D12, D22, D66 = _D_components(D)

    # Work with the short dimension as b
    if a < b:
        a, b = b, a

    pi2_b2 = _np.pi**2 / b**2
    D11D22 = _np.sqrt(_np.maximum(D11 * D22, 1e-30))
    eta    = (D12 + 2.0 * D66) / _np.maximum(D11D22, 1e-30)

    # Seydel approximation (conservative; valid for eta >= 0.5)
    eta_safe = _np.maximum(eta, 0.5)
    k_xy = 8.125 + 5.045 / eta_safe

    return float(pi2_b2 * k_xy * (D11 * D22**3) ** 0.25)


# ---------------------------------------------------------------------------
# Smooth versions for CasADi optimization
# ---------------------------------------------------------------------------

def Nxx_cr_smooth(D, a: float, b: float, m: int = 1):
    """
    Critical uniaxial compression — CasADi-compatible, fixed mode m.

    Uses a fixed half-wave mode number m (default 1 — conservative for
    panels with a/b > 1).  Setting m = round(a/b) gives a better estimate
    for square-ish panels.

    All D components may be CasADi MX expressions — the formula is purely
    algebraic with no branches.
    """
    D11, D12, D22, D66 = D[0, 0], D[0, 1], D[1, 1], D[2, 2]
    ratio = float(m) * b / a
    return (np.pi**2 / b**2) * (D11 * ratio**2 + 2*(D12 + 2*D66) + D22 / ratio**2)


def Nyy_cr_smooth(D, a: float, b: float, m: int = 1):
    """Critical y-compression — CasADi-compatible. Swaps (a,b) and (D11,D22)."""
    D11, D12, D22, D66 = D[0, 0], D[0, 1], D[1, 1], D[2, 2]
    ratio = float(m) * a / b   # note: a/b swapped relative to Nxx
    return (np.pi**2 / a**2) * (D22 * ratio**2 + 2*(D12 + 2*D66) + D11 / ratio**2)


def Nxy_cr_smooth(D, a: float, b: float, eps: float = 1e-30):
    """
    Critical shear load — CasADi-compatible Seydel approximation.

    Uses the smaller dimension as b to ensure a/b >= 1.
    eps prevents division by zero when D11*D22 ~ 0.
    """
    D11, D12, D22, D66 = D[0, 0], D[0, 1], D[1, 1], D[2, 2]
    b_eff = min(a, b)
    pi2_b2 = np.pi**2 / b_eff**2
    D11D22 = np.sqrt(D11 * D22 + eps)
    eta    = (D12 + 2.0 * D66) / (D11D22 + eps)
    k_xy   = 8.125 + 5.045 / (eta + 0.5 / (eta + eps))   # smooth softmax-style floor
    return pi2_b2 * k_xy * (D11 * D22**3 + eps) ** 0.25


# ---------------------------------------------------------------------------
# Combined buckling reserve factor
# ---------------------------------------------------------------------------

def buckling_rf(
    N_applied: _np.ndarray,
    D:         _np.ndarray,
    a:         float,
    b:         float,
    n_modes:   int = 8,
) -> float:
    """
    Combined buckling RF under Nxx + Nyy + Nxy using linear interaction.

    Interaction formula (Whitney 1987):
        R_x + R_y + R_s^2 = 1  at buckling
        where R_x = |Nxx|/N_xcr, R_y = |Nyy|/N_ycr, R_s = |Nxy|/N_xycr

    Rearranging for the reserve factor RF (load scaling to failure):
        (R_x + R_y)*RF + R_s^2*RF^2 = 1
        R_s^2*RF^2 + (R_x + R_y)*RF - 1 = 0
        RF = [ -(R_x+R_y) + sqrt((R_x+R_y)^2 + 4*R_s^2) ] / (2*R_s^2)

    For R_s = 0 (pure compression):  RF = 1 / (R_x + R_y)
    For R_x = R_y = 0 (pure shear):  RF = 1 / R_s

    Parameters
    ----------
    N_applied : [Nxx, Nyy, Nxy]  applied running loads [N/m]
    D         : (3,3) bending stiffness [N.m]
    a, b      : panel spanwise and chordwise dimensions [m]
    n_modes   : modes to evaluate for Nxx/Nyy critical loads

    Returns
    -------
    RF_buckle : float  reserve factor (< 1 = buckled)
    """
    _check_bend_twist_coupling(D)
    Nxx, Nyy, Nxy = float(N_applied[0]), float(N_applied[1]), float(N_applied[2])

    Ncx  = Nxx_cr(D, a, b, n_modes)
    Ncy  = Nyy_cr(D, a, b, n_modes)
    Ncxy = Nxy_cr(D, a, b)

    # Only compressive loads drive compression buckling
    Rx  = max(-Nxx, 0.0) / _np.maximum(Ncx,  1.0)
    Ry  = max(-Nyy, 0.0) / _np.maximum(Ncy,  1.0)
    Rs  = abs(Nxy)        / _np.maximum(Ncxy, 1.0)

    lin = Rx + Ry
    if lin < 1e-12 and Rs < 1e-12:
        return _np.inf   # essentially no load

    if Rs < 1e-12:
        return 1.0 / lin   # pure compression case

    # Quadratic: Rs^2 * RF^2 + lin * RF - 1 = 0
    disc = lin**2 + 4.0 * Rs**2
    RF   = (-lin + _np.sqrt(disc)) / (2.0 * Rs**2)
    return float(RF)


def buckling_rf_smooth(
    Nxx, Nyy, Nxy,
    D,
    a:   float,
    b:   float,
    m_x: int = 1,
    m_y: int = 1,
    eps: float = 1e-30,
):
    """
    Combined buckling RF — fully CasADi-compatible for use inside opti.

    Uses smooth Seydel shear formula and fixed mode numbers m_x, m_y.
    All load and D arguments may be CasADi MX expressions.

    Interaction: R_s^2*RF^2 + (R_x + R_y)*RF - 1 = 0
    RF = (-(R_x+R_y) + sqrt((R_x+R_y)^2 + 4*R_s^2)) / (2*R_s^2 + eps)

    Parameters
    ----------
    Nxx, Nyy, Nxy : applied running loads [N/m]  (CasADi or float)
    D             : (3,3) bending stiffness — CasADi or numpy
    a, b          : panel dimensions [m]  (float constants)
    m_x, m_y      : buckling mode numbers for Nxx and Nyy directions
    eps           : small regularisation constant

    Returns
    -------
    RF_buckle : CasADi or float reserve factor
    """
    Ncx  = Nxx_cr_smooth(D, a, b, m_x)
    Ncy  = Nyy_cr_smooth(D, a, b, m_y)
    Ncxy = Nxy_cr_smooth(D, a, b, eps)

    # Compressive loads only (smooth relu: max(x,0) ~ softplus, but here just use x
    # clipped at 0; for optimizer the constraint RF>=1 makes sense only when loading
    # is compressive, so we use abs for Nxy and negated for compression)
    Rx  = np.fmax(-Nxx, 0.0) / (Ncx  + eps)
    Ry  = np.fmax(-Nyy, 0.0) / (Ncy  + eps)
    Rs  = np.sqrt(Nxy**2 + eps) / (Ncxy + eps)

    lin  = Rx + Ry
    disc = lin**2 + 4.0 * Rs**2
    RF   = (-lin + np.sqrt(disc + eps)) / (2.0 * Rs**2 + eps)
    return RF


# ---------------------------------------------------------------------------
# Panel geometry helper
# ---------------------------------------------------------------------------

def suggest_mode_number(a: float, b: float, D: _np.ndarray) -> int:
    """
    Suggest the half-wave mode number m* that minimises N_xcr.

    For isotropic plates: m* = round(a/b).
    For orthotropic plates: m* depends on (D11/D22)^(1/4) * a/b.
    This function evaluates modes 1..10 and returns the minimising m.
    """
    D11, D12, D22, D66 = _D_components(D)
    two_D = 2.0 * (D12 + 2.0 * D66)
    m = _np.arange(1, 11, dtype=float)
    r = m * b / a
    N = D11 * r**2 + two_D + D22 / r**2
    return int(m[_np.argmin(N)])


if __name__ == "__main__":
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
    from composite_panel.ply import Ply, IM7_8552
    from composite_panel.laminate import Laminate

    mat   = IM7_8552()
    t_ply = 0.125e-3   # 0.125 mm cured-ply thickness

    # [0/45/-45/90]s quasi-isotropic symmetric laminate (8 plies, h = 1 mm)
    angles = [0, 45, -45, 90, 90, -45, 45, 0]
    plies  = [Ply(mat, t_ply, a) for a in angles]
    lam    = Laminate(plies)
    D      = lam.D

    # Rib/stringer bay: 0.5 m spanwise, 0.15 m chordwise
    a, b = 0.5, 0.15

    print(f"Panel {a:.2f} m (span) x {b:.2f} m (chord),  h = {lam.thickness*1e3:.2f} mm")
    print(f"  Nxx_cr = {Nxx_cr(D, a, b)/1e3:.1f} kN/m")
    print(f"  Nyy_cr = {Nyy_cr(D, a, b)/1e3:.1f} kN/m")
    print(f"  Nxy_cr = {Nxy_cr(D, a, b)/1e3:.1f} kN/m")
    print(f"  Suggested mode m* = {suggest_mode_number(a, b, D)}")
    print()

    # Applied loads: upper-skin compression + shear
    N_applied = _np.array([-280e3, -115e3, 42e3])  # [Nxx, Nyy, Nxy] N/m
    rf = buckling_rf(N_applied, D, a, b)
    print(f"Applied: Nxx={N_applied[0]/1e3:.0f}, Nyy={N_applied[1]/1e3:.0f}, "
          f"Nxy={N_applied[2]/1e3:.0f} kN/m")
    print(f"Buckling RF = {rf:.3f}  ({'OK' if rf >= 1.0 else 'BUCKLED'})")
