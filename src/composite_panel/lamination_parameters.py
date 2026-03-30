"""
composite_panel.lamination_parameters
--------------------------------------
Lamination parameters (LPs) and stiffness polar analysis for composite laminates.

THEORY
======
Lamination parameters are dimensionless integrals of trigonometric functions of
the ply angle through the laminate thickness.  They characterise the stiffness
completely for any layup with the same material, and span a convex feasible
domain — making them ideal for continuous gradient-based optimisation.

For a laminate of total thickness h, the four in-plane LPs are:

    ξ1A = (1/h) ∫ cos(2θ) dz
    ξ2A = (1/h) ∫ cos(4θ) dz
    ξ3A = (1/h) ∫ sin(2θ) dz   ← zero for balanced laminates (±θ pairs)
    ξ4A = (1/h) ∫ sin(4θ) dz   ← zero for balanced laminates

And the four bending LPs (ξ1D..ξ4D) use the same integrands weighted by 12z²/h³.
Coupling LPs (ξ1B..ξ4B) are weighted by 4z/h² and vanish for symmetric laminates.

The ABD matrices are linear in the LPs via the Tsai-Pagano material invariants
U1..U5:

    A11 = h · (U1 + U2·ξ1A + U3·ξ2A)
    A22 = h · (U1 − U2·ξ1A + U3·ξ2A)
    A12 = h · (U4 − U3·ξ2A)
    A66 = h · (U5 − U3·ξ2A)
    A16 = h · (U2/2·ξ3A + U3·ξ4A)   ← non-zero for unbalanced laminates
    A26 = h · (U2/2·ξ3A − U3·ξ4A)   ← non-zero for unbalanced laminates

And identically for D (replacing h with h³/12 and ξA with ξD).

STIFFNESS POLARS
================
For a single-angle laminate [θ]_n, the LPs reduce to:
    ξ1A = cos(2θ),  ξ2A = cos(4θ),  ξ3A = sin(2θ),  ξ4A = sin(4θ)

Sweeping θ from 0° to 180° produces stiffness polar plots showing how Ex, Ey,
Gxy (in-plane) and D11, D22, D66 (bending) vary with fibre orientation — useful
for inferring optimal ply angles for a given load state without running a full
optimisation.

AEROELASTIC TAILORING CONNECTION
=================================
Non-zero A16/A26 (unbalanced) and D16/D26 (unbalanced or off-axis stacking)
introduce shear-extension and bend-twist coupling respectively.  For a swept
wing, D16/D26 couple wing bending to chordwise twist, enabling passive
aeroelastic wash-out (load relief) or wash-in (load amplification) to be
engineered purely through laminate design — without geometric changes.

References
----------
Tsai, S.W. & Hahn, H.T. — Introduction to Composite Materials (1980), Ch. 9
Miki, M. — Material design of composite laminates with required in-plane
    elastic properties (1982) ICCM-4 progress in science and engineering of
    composites, pp. 1725-1731.
Gürdal, Z., Haftka, R.T. & Hajela, P. — Design and Optimization of Laminated
    Composite Materials (Wiley, 1999), Ch. 4
"""

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np

from .ply import PlyMaterial, Ply

if TYPE_CHECKING:
    from .laminate import Laminate


# ---------------------------------------------------------------------------
# Material invariants (Tsai-Pagano)
# ---------------------------------------------------------------------------

def material_invariants(mat: PlyMaterial) -> dict:
    """
    Compute the five Tsai-Pagano material invariants U1..U5.

    These are angle-independent combinations of the reduced stiffness
    coefficients Qij.  They act as the "anchor points" from which the
    laminate stiffness can be reconstructed purely from the lamination
    parameters.

    Parameters
    ----------
    mat : PlyMaterial
        Orthotropic ply material (E1, E2, G12, nu12).

    Returns
    -------
    dict with keys 'U1' through 'U5' [Pa].

    Notes
    -----
    Derivation follows from integrating Q̄(θ) over a complete revolution,
    which eliminates all angle-dependent terms:

        U1 = (3Q11 + 3Q22 + 2Q12 + 4Q66) / 8
        U2 = (Q11 − Q22) / 2
        U3 = (Q11 + Q22 − 2Q12 − 4Q66) / 8
        U4 = (Q11 + Q22 + 6Q12 − 4Q66) / 8
        U5 = (Q11 + Q22 − 2Q12 + 4Q66) / 8  =  (U1 − U4)

    Quick sanity check: for an isotropic material (E1=E2, nu12=nu21, G12=E/(2+2nu))
    U2 = U3 = 0, and U1 = U4 + U5.
    """
    denom = 1.0 - mat.nu12 * mat.nu21
    Q11 = mat.E1  / denom
    Q22 = mat.E2  / denom
    Q12 = mat.nu12 * mat.E2 / denom
    Q66 = mat.G12

    U1 = (3*Q11 + 3*Q22 + 2*Q12 + 4*Q66) / 8.0
    U2 = (Q11 - Q22) / 2.0
    U3 = (Q11 + Q22 - 2*Q12 - 4*Q66) / 8.0
    U4 = (Q11 + Q22 + 6*Q12 - 4*Q66) / 8.0
    U5 = (Q11 + Q22 - 2*Q12 + 4*Q66) / 8.0   # = U1 - U4

    return {'U1': U1, 'U2': U2, 'U3': U3, 'U4': U4, 'U5': U5,
            'Q11': Q11, 'Q22': Q22, 'Q12': Q12, 'Q66': Q66}


# ---------------------------------------------------------------------------
# LP computation from a discrete laminate
# ---------------------------------------------------------------------------

def lamination_parameters(lam: 'Laminate') -> dict:
    """
    Compute the full set of 12 lamination parameters for a discrete laminate.

    Parameters
    ----------
    lam : Laminate
        Assembled Laminate object (any layup, symmetric or not).

    Returns
    -------
    dict with keys:
        'xi1A'..'xi4A'  — in-plane LPs   (dimensionless)
        'xi1B'..'xi4B'  — coupling LPs   (dimensionless, zero for symmetric)
        'xi1D'..'xi4D'  — bending LPs    (dimensionless)

    Notes
    -----
    The integration is performed numerically using the ply midplane angles
    and the standard thickness weighting:

        ξiA = (1/h) Σ_k  fi(θ_k) · Δz_k
        ξiB = (4/h²) Σ_k  fi(θ_k) · z_mid_k · Δz_k
        ξiD = (12/h³) Σ_k  fi(θ_k) · z_mid_k² · Δz_k

    where fi = [cos2θ, cos4θ, sin2θ, sin4θ] for i = 1..4.

    For a balanced laminate (equal ±θ pairs): ξ3A = ξ4A = 0.
    For a symmetric laminate (mirror about mid-plane): ξ1B..ξ4B = 0.
    """
    h     = lam.thickness
    z0s   = lam.z_interfaces[:-1]
    z1s   = lam.z_interfaces[1:]
    dz    = z1s - z0s
    z_mid = (z0s + z1s) / 2.0

    # Ply angles [rad]
    angles_rad = np.array([math.radians(p.angle_deg) for p in lam.plies])

    # Trigonometric basis functions fi(θ)
    cos2 = np.cos(2.0 * angles_rad)
    cos4 = np.cos(4.0 * angles_rad)
    sin2 = np.sin(2.0 * angles_rad)
    sin4 = np.sin(4.0 * angles_rad)

    basis = np.vstack([cos2, cos4, sin2, sin4])   # (4, n_plies)

    # In-plane LPs: ξiA = (1/h) Σ fi(θk) * Δzk
    xi_A = (basis @ dz) / h

    # Coupling LPs: ξiB = (4/h²) Σ fi(θk) * z_mid_k * Δzk
    xi_B = (basis @ (z_mid * dz)) * (4.0 / h**2)

    # Bending LPs: ξiD = (12/h³) Σ fi(θk) * z_mid_k² * Δzk
    xi_D = (basis @ (z_mid**2 * dz)) * (12.0 / h**3)

    return {
        'xi1A': float(xi_A[0]), 'xi2A': float(xi_A[1]),
        'xi3A': float(xi_A[2]), 'xi4A': float(xi_A[3]),
        'xi1B': float(xi_B[0]), 'xi2B': float(xi_B[1]),
        'xi3B': float(xi_B[2]), 'xi4B': float(xi_B[3]),
        'xi1D': float(xi_D[0]), 'xi2D': float(xi_D[1]),
        'xi3D': float(xi_D[2]), 'xi4D': float(xi_D[3]),
    }


# ---------------------------------------------------------------------------
# ABD reconstruction from LPs
# ---------------------------------------------------------------------------

def abd_from_lamination_params(
    h:   float,
    lp:  dict,
    mat: PlyMaterial,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reconstruct the A, B, D stiffness matrices from lamination parameters.

    This is the inverse mapping from LP space back to physical stiffness —
    useful for checking feasibility and for LP-based optimisation.

    Parameters
    ----------
    h   : float
        Total laminate thickness [m].
    lp  : dict
        Lamination parameters as returned by lamination_parameters().
    mat : PlyMaterial
        Ply material (needed for invariants U1..U5).

    Returns
    -------
    A : np.ndarray (3, 3)  [N/m]
    B : np.ndarray (3, 3)  [N]
    D : np.ndarray (3, 3)  [N·m]

    Notes
    -----
    The reconstruction is exact for any laminate built from a single
    material system.  The formulas are:

        A11 = h·(U1 + U2·ξ1A + U3·ξ2A)
        A22 = h·(U1 − U2·ξ1A + U3·ξ2A)
        A12 = h·(U4 − U3·ξ2A)
        A66 = h·(U5 − U3·ξ2A)
        A16 = h·(U2/2·ξ3A + U3·ξ4A)
        A26 = h·(U2/2·ξ3A − U3·ξ4A)

    and identically for D (h → h³/12, ξA → ξD) and B (h → h²/4, ξA → ξB).
    """
    inv = material_invariants(mat)
    U1, U2, U3, U4, U5 = inv['U1'], inv['U2'], inv['U3'], inv['U4'], inv['U5']

    def _stiffness_3x3(scale, x1, x2, x3, x4):
        s11 = scale * (U1 + U2*x1 + U3*x2)
        s22 = scale * (U1 - U2*x1 + U3*x2)
        s12 = scale * (U4 - U3*x2)
        s66 = scale * (U5 - U3*x2)
        s16 = scale * (0.5*U2*x3 + U3*x4)
        s26 = scale * (0.5*U2*x3 - U3*x4)
        return np.array([[s11, s12, s16],
                         [s12, s22, s26],
                         [s16, s26, s66]])

    A = _stiffness_3x3(h,        lp['xi1A'], lp['xi2A'], lp['xi3A'], lp['xi4A'])
    B = _stiffness_3x3(h**2/4,   lp['xi1B'], lp['xi2B'], lp['xi3B'], lp['xi4B'])
    D = _stiffness_3x3(h**3/12,  lp['xi1D'], lp['xi2D'], lp['xi3D'], lp['xi4D'])

    return A, B, D


# ---------------------------------------------------------------------------
# Stiffness polar plots
# ---------------------------------------------------------------------------

def stiffness_polar(
    mat:             PlyMaterial,
    h_total:         float,
    n_angles:        int   = 181,
    ax_inplane:      object = None,
    ax_bending:      object = None,
    label:           str   = '',
    show:            bool  = True,
) -> object:
    """
    Plot in-plane and bending stiffness as a function of single-angle fibre
    orientation θ (0° → 180°).

    For a single-angle laminate [θ]_n the lamination parameters collapse to:
        ξ1A = cos(2θ),  ξ2A = cos(4θ),  ξ3A = sin(2θ),  ξ4A = sin(4θ)

    This function sweeps θ and plots:
        In-plane  : Ex(θ), Ey(θ), Gxy(θ)  [GPa]
        Bending   : D11(θ), D22(θ), D66(θ) normalised by D11(0°)

    Parameters
    ----------
    mat       : PlyMaterial
        Ply material.
    h_total   : float
        Total laminate thickness [m].
    n_angles  : int
        Number of angle points to sweep (default 181 → 1° steps).
    ax_inplane : matplotlib Axes, optional
        Axes to draw in-plane plot onto.  If None, created internally.
    ax_bending : matplotlib Axes, optional
        Axes to draw bending plot onto.  If None, created internally.
    label     : str
        Legend label suffix (useful when overlaying multiple materials).
    show      : bool
        Call plt.show() at the end (set False for embedding in larger figures).

    Returns
    -------
    fig : matplotlib Figure
        Figure containing both subplots.

    Notes
    -----
    The stiffness polars are a fast way to infer optimal ply orientations
    without running a full NLP:

      • High axial load Nxx   → maximise Ex  → θ ≈ 0°
      • High transverse Nyy   → maximise Ey  → θ ≈ 90°
      • High shear Nxy        → maximise Gxy → θ ≈ 45°
      • Bending Mxx, buckling → maximise D11 → θ ≈ 0°
      • Twisting Mxy          → maximise D66 → θ ≈ 45°

    The crossover angles (where Ex = Ey, or D11 = D22) reveal the
    quasi-isotropic orientation and guide multi-objective trade-offs.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    from .laminate import Laminate

    inv = material_invariants(mat)
    U1, U2, U3, U4, U5 = inv['U1'], inv['U2'], inv['U3'], inv['U4'], inv['U5']
    h = h_total

    thetas_deg = np.linspace(0, 180, n_angles)
    thetas_rad = np.radians(thetas_deg)

    # Single-angle LP values
    cos2 = np.cos(2 * thetas_rad)
    cos4 = np.cos(4 * thetas_rad)
    sin2 = np.sin(2 * thetas_rad)
    sin4 = np.sin(4 * thetas_rad)

    # In-plane stiffness [Pa]
    A11 = h * (U1 + U2*cos2 + U3*cos4)
    A22 = h * (U1 - U2*cos2 + U3*cos4)
    A12 = h * (U4 - U3*cos4)
    A66 = h * (U5 - U3*cos4)
    A16 = h * (0.5*U2*sin2 + U3*sin4)
    A26 = h * (0.5*U2*sin2 - U3*sin4)

    # Engineering constants from compliance (A16 != 0 for off-axis)
    # Full 3×3 inversion per angle is accurate for the off-axis case
    Ex_arr  = np.empty(n_angles)
    Ey_arr  = np.empty(n_angles)
    Gxy_arr = np.empty(n_angles)

    for i in range(n_angles):
        A_mat = np.array([[A11[i], A12[i], A16[i]],
                          [A12[i], A22[i], A26[i]],
                          [A16[i], A26[i], A66[i]]])
        try:
            a_mat = np.linalg.inv(A_mat)
            Ex_arr[i]  = 1.0 / (a_mat[0, 0] * h)
            Ey_arr[i]  = 1.0 / (a_mat[1, 1] * h)
            Gxy_arr[i] = 1.0 / (a_mat[2, 2] * h)
        except np.linalg.LinAlgError:
            Ex_arr[i] = Ey_arr[i] = Gxy_arr[i] = float('nan')

    # Bending stiffness [N·m]
    D11 = (h**3/12) * (U1 + U2*cos2 + U3*cos4)
    D22 = (h**3/12) * (U1 - U2*cos2 + U3*cos4)
    D12 = (h**3/12) * (U4 - U3*cos4)
    D66 = (h**3/12) * (U5 - U3*cos4)
    D16 = (h**3/12) * (0.5*U2*sin2 + U3*sin4)
    D26 = (h**3/12) * (0.5*U2*sin2 - U3*sin4)

    D11_ref = float(D11[0])   # normalise to D11 at θ=0°

    # ── Plotting ─────────────────────────────────────────────────────────────
    if ax_inplane is None or ax_bending is None:
        fig, (ax_in, ax_bd) = plt.subplots(1, 2, figsize=(12, 5),
                                            facecolor='#0d1117')
        for ax in (ax_in, ax_bd):
            ax.set_facecolor('#161b22')
            for spine in ax.spines.values():
                spine.set_color('#30363d')
            ax.tick_params(colors='#8b949e', labelsize=9)
            ax.xaxis.label.set_color('#c9d1d9')
            ax.yaxis.label.set_color('#c9d1d9')
            ax.title.set_color('#f0f6fc')
    else:
        ax_in, ax_bd = ax_inplane, ax_bending
        fig = ax_in.get_figure()

    lbl = f' ({label})' if label else ''
    GPa = 1e9

    # In-plane
    ax_in.plot(thetas_deg, Ex_arr  / GPa, color='#58a6ff', lw=2,
               label=f'$E_x${lbl}')
    ax_in.plot(thetas_deg, Ey_arr  / GPa, color='#f78166', lw=2,
               label=f'$E_y${lbl}')
    ax_in.plot(thetas_deg, Gxy_arr / GPa, color='#3fb950', lw=2,
               label=f'$G_{{xy}}${lbl}')
    ax_in.set_xlabel('Fibre angle θ [°]')
    ax_in.set_ylabel('Effective modulus [GPa]')
    ax_in.set_title('In-plane stiffness polar')
    ax_in.set_xlim(0, 180)
    ax_in.set_xticks(range(0, 181, 30))
    ax_in.legend(framealpha=0.25, labelcolor='#c9d1d9', fontsize=9)
    ax_in.grid(True, color='#21262d', lw=0.5)

    # Bending — normalised to D11(0°) for unit-less comparison
    ax_bd.plot(thetas_deg, D11 / D11_ref, color='#58a6ff', lw=2,
               label=f'$D_{{11}}/D_{{11}}^0${lbl}')
    ax_bd.plot(thetas_deg, D22 / D11_ref, color='#f78166', lw=2,
               label=f'$D_{{22}}/D_{{11}}^0${lbl}')
    ax_bd.plot(thetas_deg, D66 / D11_ref, color='#3fb950', lw=2,
               label=f'$D_{{66}}/D_{{11}}^0${lbl}')
    ax_bd.plot(thetas_deg, np.abs(D16) / D11_ref, color='#d2a8ff', lw=1.5,
               ls='--', label=f'$|D_{{16}}|/D_{{11}}^0${lbl}')
    ax_bd.plot(thetas_deg, np.abs(D26) / D11_ref, color='#ffa657', lw=1.5,
               ls='--', label=f'$|D_{{26}}|/D_{{11}}^0${lbl}')
    ax_bd.set_xlabel('Fibre angle θ [°]')
    ax_bd.set_ylabel(f'Bending stiffness / $D_{{11}}(0°)$')
    ax_bd.set_title('Out-of-plane (bending) stiffness polar')
    ax_bd.set_xlim(0, 180)
    ax_bd.set_xticks(range(0, 181, 30))
    ax_bd.legend(framealpha=0.25, labelcolor='#c9d1d9', fontsize=9)
    ax_bd.grid(True, color='#21262d', lw=0.5)

    # Annotate peak locations
    for ax, arrs, names in [
        (ax_in,
         [Ex_arr/GPa, Ey_arr/GPa, Gxy_arr/GPa],
         ['$E_x$', '$E_y$', '$G_{xy}$']),
        (ax_bd,
         [D11/D11_ref, D22/D11_ref, D66/D11_ref],
         ['$D_{11}$', '$D_{22}$', '$D_{66}$']),
    ]:
        for arr, name in zip(arrs, names):
            idx_peak = int(np.argmax(arr))
            ax.axvline(thetas_deg[idx_peak], color='#ffffff', lw=0.5, alpha=0.3)

    fig.suptitle(
        f'Stiffness polars — {getattr(mat, "name", "composite")}  |  '
        f'h = {h_total*1e3:.2f} mm',
        color='#f0f6fc', fontsize=11,
    )
    fig.tight_layout()

    if show:
        plt.show()

    return fig


# ---------------------------------------------------------------------------
# LP feasibility domain (in-plane, ξ1A–ξ2A plane)
# ---------------------------------------------------------------------------

def plot_lp_feasibility(ax=None, show: bool = True) -> object:
    """
    Sketch the feasible domain for in-plane lamination parameters in the
    (ξ1A, ξ2A) plane.

    The physical constraints are:
        −1 ≤ ξ1A ≤ 1
        2ξ1A² − 1 ≤ ξ2A ≤ 1     (Miki's parabolic boundary)

    The parabolic lower bound arises from the Cauchy-Schwarz inequality
    applied to the LP integrals: cos(4θ) ≥ 2cos²(2θ) − 1.

    Single-angle laminates trace the boundary parabola (ξ2A = 2ξ1A² − 1).
    Cross-plies and angle-plies populate the interior.

    Parameters
    ----------
    ax   : matplotlib Axes, optional
    show : bool

    Returns
    -------
    fig : matplotlib Figure
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.path import Path

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5), facecolor='#0d1117')
        ax.set_facecolor('#161b22')
        for spine in ax.spines.values():
            spine.set_color('#30363d')
        ax.tick_params(colors='#8b949e', labelsize=9)
    else:
        fig = ax.get_figure()

    # Feasibility boundary
    xi1 = np.linspace(-1, 1, 400)
    xi2_lo = 2*xi1**2 - 1   # Miki's parabola (lower)
    xi2_hi = np.ones_like(xi1)

    # Shade feasible region
    ax.fill_between(xi1, xi2_lo, xi2_hi, alpha=0.15, color='#58a6ff',
                    label='Feasible domain')
    ax.plot(xi1, xi2_lo, color='#58a6ff', lw=1.5, ls='--',
            label=r'$\xi_{2A} = 2\xi_{1A}^2 - 1$ (single angle)')
    ax.axhline(1, color='#58a6ff', lw=1.5, ls='--')

    # Annotate characteristic points
    pts = {
        '0°':   (1, 1),
        '90°':  (-1, 1),
        '±45°': (0, -1),
        'QI':   (0, 0),   # quasi-isotropic
    }
    for name, (x, y) in pts.items():
        ax.scatter(x, y, s=60, color='#f78166', zorder=5)
        ax.annotate(name, (x, y), textcoords='offset points', xytext=(6, 4),
                    color='#c9d1d9', fontsize=8)

    ax.set_xlabel(r'$\xi_{1A}$', color='#c9d1d9')
    ax.set_ylabel(r'$\xi_{2A}$', color='#c9d1d9')
    ax.set_title('LP feasibility domain (in-plane)', color='#f0f6fc')
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.legend(framealpha=0.25, labelcolor='#c9d1d9', fontsize=8)
    ax.grid(True, color='#21262d', lw=0.5)
    ax.set_aspect('equal')

    if show:
        plt.show()

    return fig


# ---------------------------------------------------------------------------
# Unbalanced laminate coupling metrics
# ---------------------------------------------------------------------------

def bend_twist_coupling_index(lam: 'Laminate') -> float:
    """
    Return a dimensionless bend-twist coupling index for the laminate.

    Defined as:

        BTC = |D16| / sqrt(D11 · D66)

    BTC = 0   for balanced laminates (D16 = 0)
    BTC → 1   for maximum coupling (theoretical upper bound)

    This index quantifies the aeroelastic tailoring potential:
    high BTC with appropriate sign gives effective wash-out on swept wings.

    Parameters
    ----------
    lam : Laminate

    Returns
    -------
    float in [0, 1]
    """
    D = lam.D
    D16 = D[0, 2]
    D11 = D[0, 0]
    D66 = D[2, 2]
    denom = math.sqrt(abs(D11 * D66))
    if denom < 1e-30:
        return 0.0
    return abs(D16) / denom


def shear_extension_coupling_index(lam: 'Laminate') -> float:
    """
    Dimensionless shear-extension coupling index.

        SEC = |A16| / sqrt(A11 · A66)

    SEC = 0 for balanced laminates (A16 = 0).
    Non-zero SEC causes in-plane shear distortion under pure axial loading,
    which is relevant for aeroelastic tailoring of compression-loaded panels.

    Parameters
    ----------
    lam : Laminate

    Returns
    -------
    float in [0, 1]
    """
    A = lam.A
    A16 = A[0, 2]
    A11 = A[0, 0]
    A66 = A[2, 2]
    denom = math.sqrt(abs(A11 * A66))
    if denom < 1e-30:
        return 0.0
    return abs(A16) / denom
