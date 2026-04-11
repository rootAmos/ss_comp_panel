"""
Panel buckling under combined Nxx + Nyy + Nxy.

Simply-supported orthotropic plate, Navier solution for compression,
Seydel approximation for shear, Whitney (1987) interaction.

Ref: Timoshenko & Gere (1961) Ch. 9, Whitney (1987), ESDU 02.03.11
"""

from __future__ import annotations

import warnings
import numpy as _np
from typing import Optional, Tuple, Union

import aerosandbox.numpy as np


def _D_components(D: _np.ndarray) -> Tuple:
    """Extract D11, D12, D22, D66."""
    return D[0, 0], D[0, 1], D[1, 1], D[2, 2]


def _check_bend_twist_coupling(D: _np.ndarray, tol: float = 0.05) -> None:
    """Warn if D16/D26 are significant -- orthotropic formulas overpredict Ncr."""
    D11, D22 = float(D[0, 0]), float(D[1, 1])
    D16, D26 = float(D[0, 2]), float(D[1, 2])
    ref = _np.sqrt(_np.maximum(D11 * D22, 1e-30))
    ratio = max(abs(D16), abs(D26)) / ref
    if ratio > tol:
        warnings.warn(
            f"Significant bend-twist coupling: "
            f"max(|D16|,|D26|)/sqrt(D11*D22) = {ratio:.3f} > {tol:.2f}. "
            f"Orthotropic buckling formula will overpredict Ncr.",
            stacklevel=3,
        )


# ---------------------------------------------------------------------------
# Critical loads -- numpy (exact, post-processing)
# ---------------------------------------------------------------------------

def Nxx_cr(D: _np.ndarray, a: float, b: float, n_modes: int = 8) -> float:
    """SS orthotropic plate, uniaxial Nxx.  Timoshenko & Gere (1961), Eq. 9.7; Whitney (1987), Eq. 5.1."""
    D11, D12, D22, D66 = _D_components(D)
    pi2_b2 = _np.pi**2 / b**2
    two_D12_D66 = 2.0 * (D12 + 2.0 * D66)

    m = _np.arange(1, n_modes + 1, dtype=float)
    ratios = m * b / a
    loads = pi2_b2 * (D11 * ratios**2 + two_D12_D66 + D22 / ratios**2)
    return float(_np.min(loads))


def Nyy_cr(D: _np.ndarray, a: float, b: float, n_modes: int = 8) -> float:
    """Chordwise critical load [N/m].  Swaps D11<->D22 and a<->b."""
    D11, D12, D22, D66 = D[0, 0], D[0, 1], D[1, 1], D[2, 2]
    D_swapped = _np.array([
        [D22, D12,   0],
        [D12, D11,   0],
        [  0,   0, D66],
    ])
    return Nxx_cr(D_swapped, b, a, n_modes)


def Nxy_cr(D: _np.ndarray, a: float, b: float) -> float:
    """Shear buckling [N/m].  Whitney (1987), Sec. 5.2, fit to Seydel (1933) charts.
    k_xy = 8.125 + 5.045/eta.  ~5% for 0.5 < eta < 5."""
    D11, D12, D22, D66 = _D_components(D)
    if a < b:
        a, b = b, a

    pi2_b2 = _np.pi**2 / b**2
    D11D22 = _np.sqrt(_np.maximum(D11 * D22, 1e-30))
    eta    = (D12 + 2.0 * D66) / _np.maximum(D11D22, 1e-30)
    eta_safe = _np.maximum(eta, 0.5)
    k_xy = 8.125 + 5.045 / eta_safe

    return float(pi2_b2 * k_xy * (D11 * D22**3) ** 0.25)


# ---------------------------------------------------------------------------
# Smooth versions for CasADi
# ---------------------------------------------------------------------------

def Nxx_cr_smooth(D, a: float, b: float, m: int = 1):
    """CasADi-compatible Nxx_cr with fixed mode number m."""
    D11, D12, D22, D66 = D[0, 0], D[0, 1], D[1, 1], D[2, 2]
    ratio = float(m) * b / a
    return (np.pi**2 / b**2) * (D11 * ratio**2 + 2*(D12 + 2*D66) + D22 / ratio**2)


def Nyy_cr_smooth(D, a: float, b: float, m: int = 1):
    """CasADi-compatible Nyy_cr."""
    D11, D12, D22, D66 = D[0, 0], D[0, 1], D[1, 1], D[2, 2]
    ratio = float(m) * a / b
    return (np.pi**2 / a**2) * (D22 * ratio**2 + 2*(D12 + 2*D66) + D11 / ratio**2)


def Nxy_cr_smooth(D, a: float, b: float, eps: float = 1e-30):
    """CasADi-compatible Seydel shear buckling."""
    D11, D12, D22, D66 = D[0, 0], D[0, 1], D[1, 1], D[2, 2]
    b_eff = min(a, b)
    pi2_b2 = np.pi**2 / b_eff**2
    D11D22 = np.sqrt(D11 * D22 + eps)
    eta    = (D12 + 2.0 * D66) / (D11D22 + eps)
    k_xy   = 8.125 + 5.045 / np.fmax(eta, 0.5)
    return pi2_b2 * k_xy * (D11 * D22**3 + eps) ** 0.25


# ---------------------------------------------------------------------------
# Rayleigh-Ritz anisotropic buckling (accounts for D16/D26)
# ---------------------------------------------------------------------------
#
# D16/D26 bend-twist coupling couples modes (m,n) and (m',n') only when
# BOTH (m+m') AND (n+n') are odd.  A single-n basis gives zero coupling
# because int_0^b sin(n*pi*y/b)*cos(n*pi*y/b) dy = 0.
# We therefore use a 2-D basis with m=1..M, n=1..N.
#
# Ref: Nemeth, NASA/TP-2003-212131 (2003), Sec. 3;
#      Baucke & Mittelstedt, Composite Structures (2015), Sec. 3.

def _rr_coupling_integral(p: int, q: int, L: float) -> float:
    """int_0^L sin(p*pi*x/L) * cos(q*pi*x/L) dx.
    = 2*L*p / (pi*(p^2 - q^2))  when p+q is odd, else 0."""
    if (p + q) % 2 == 0:
        return 0.0
    return 2.0 * L * p / (_np.pi * (p**2 - q**2))


def _rr_Nxx_cr(D, a: float, b: float, M: int = 5, N: int = 2) -> float:
    """Rayleigh-Ritz Nxx_cr for anisotropic SS plate (full D, includes D16/D26).

    Basis: w = sum A_{mn} sin(m*pi*x/a) sin(n*pi*y/b), m=1..M, n=1..N.
    Generalised eigenvalue: K*A = lambda*G*A.  G diagonal for uniaxial Nxx.
    Nemeth (2003), NASA/TP-2003-212131, Eqs. 3-8.

    Returns Ncr [N/m] (positive = compression).
    """
    D11, D12, D22, D66 = float(D[0, 0]), float(D[0, 1]), float(D[1, 1]), float(D[2, 2])
    D16, D26           = float(D[0, 2]), float(D[1, 2])

    # Mode index pairs (m, n)
    modes = [(m, n) for m in range(1, M + 1) for n in range(1, N + 1)]
    P = len(modes)

    K_mat = _np.zeros((P, P))
    G_vec = _np.zeros(P)

    for i, (mi, ni) in enumerate(modes):
        K_mat[i, i] = (a * b / 4.0) * _np.pi**4 * (
            D11 * (mi / a)**4
            + 2.0 * (D12 + 2.0 * D66) * (mi / a)**2 * (ni / b)**2
            + D22 * (ni / b)**4
        )
        G_vec[i] = (a * b / 4.0) * _np.pi**2 * (mi / a)**2

        for j, (mj, nj) in enumerate(modes):
            if j <= i:
                continue

            Ix = _rr_coupling_integral(mi, mj, a)
            Iy = _rr_coupling_integral(ni, nj, b)

            if abs(Ix * Iy) < 1e-30:
                continue

            # Off-diagonal from D16 (w,xx * w,xy) and D26 (w,yy * w,xy) energy terms.
            # Each has two cross products (i,j) and (j,i); combining with
            # Ix(p,q) = -(q/p)*Ix(q,p) antisymmetry gives the symmetric form:
            #
            #   K_ij = -2*pi^4*Ix*Iy*mj*nj * [D16*(mi^2+mj^2)/(a^3*b)
            #                                 + D26*(ni^2+nj^2)/(a*b^3)]
            #
            # Derivation: Nemeth (2003), NASA/TP-2003-212131, Eqs. 3-8.
            K_ij = -2.0 * _np.pi**4 * Ix * Iy * mj * nj * (
                D16 * (mi**2 + mj**2) / (a**3 * b)
                + D26 * (ni**2 + nj**2) / (a * b**3)
            )

            K_mat[i, j] = K_ij
            K_mat[j, i] = K_ij

    # Solve: K v = lambda G v.  G diagonal → H = G^{-1} K, eigenvalues of H.
    H = K_mat / G_vec[:, None]
    eigs = _np.linalg.eigvalsh(H)
    pos_eigs = eigs[eigs > 1e-6]
    if len(pos_eigs) == 0:
        return float(_np.min(_np.abs(eigs)))
    return float(_np.min(pos_eigs))


def _rr_Nyy_cr(D, a: float, b: float, M: int = 5, N: int = 2) -> float:
    """Rayleigh-Ritz Nyy_cr — swap x<->y axes.
    Nemeth (2003), applied with coordinate swap."""
    D11, D12, D22, D66 = float(D[0, 0]), float(D[0, 1]), float(D[1, 1]), float(D[2, 2])
    D16, D26           = float(D[0, 2]), float(D[1, 2])
    D_swap = _np.array([
        [D22, D12, D26],
        [D12, D11, D16],
        [D26, D16, D66],
    ])
    return _rr_Nxx_cr(D_swap, b, a, M, N)


# ---------------------------------------------------------------------------
# Combined buckling RF
# ---------------------------------------------------------------------------

def buckling_rf(N_applied: _np.ndarray, D: _np.ndarray,
                a: float, b: float, n_modes: int = 8) -> float:
    """Combined buckling RF via Whitney (1987) interaction: R_x + R_y + R_s^2 = 1.
    Uses Rayleigh-Ritz Ncr (M=5, N=2) that accounts for D16/D26 bend-twist coupling.
    Nemeth (2003), NASA/TP-2003-212131; Baucke & Mittelstedt (2015)."""
    _check_bend_twist_coupling(D)
    Nxx, Nyy, Nxy = float(N_applied[0]), float(N_applied[1]), float(N_applied[2])

    Ncx  = _rr_Nxx_cr(D, a, b)
    Ncy  = _rr_Nyy_cr(D, a, b)
    Ncxy = Nxy_cr(D, a, b)

    Rx  = max(-Nxx, 0.0) / _np.maximum(Ncx,  1.0)
    Ry  = max(-Nyy, 0.0) / _np.maximum(Ncy,  1.0)
    Rs  = abs(Nxy)        / _np.maximum(Ncxy, 1.0)

    lin = Rx + Ry
    if lin < 1e-12 and Rs < 1e-12:
        return _np.inf

    if Rs < 1e-12:
        return 1.0 / lin

    disc = lin**2 + 4.0 * Rs**2
    RF   = (-lin + _np.sqrt(disc)) / (2.0 * Rs**2)
    return float(RF)


def _rr_Nxx_cr_smooth(D, a: float, b: float,
                       m_modes: tuple = (1, 2, 3),
                       n_modes: tuple = (1, 2),
                       eps: float = 1e-30):
    """CasADi-compatible anisotropic Nxx_cr via 2x2 closed-form eigenvalues.

    For each (m,1) base mode, D16/D26 couples it to (m±1, 2) modes. The
    minimum eigenvalue of each 2x2 coupling pair has the closed form:
        lambda_min = (h_ii + h_jj)/2 - sqrt((h_ii - h_jj)^2/4 + h_ij^2)
    which is smooth and CasADi-differentiable.

    Takes the minimum across all base modes and all coupling partners.
    Nemeth (2003), NASA/TP-2003-212131.
    """
    D11, D12, D22, D66 = D[0, 0], D[0, 1], D[1, 1], D[2, 2]
    D16, D26           = D[0, 2], D[1, 2]

    modes = [(m, n) for m in m_modes for n in n_modes]
    P = len(modes)

    def _H_diag(mi, ni):
        """Orthotropic Ncr for mode (mi, ni) = K_ii / G_ii."""
        mi_f, ni_f = float(mi), float(ni)
        K_ii = (a * b / 4.0) * np.pi**4 * (
            D11 * (mi_f / a)**4
            + 2.0 * (D12 + 2.0 * D66) * (mi_f / a)**2 * (ni_f / b)**2
            + D22 * (ni_f / b)**4
        )
        G_ii = (a * b / 4.0) * np.pi**2 * (mi_f / a)**2
        return K_ii / G_ii

    def _H_offdiag(mi, ni, mj, nj):
        """Off-diagonal H_ij = K_ij / G_ii.  Zero unless (mi+mj) and (ni+nj) both odd."""
        if (mi + mj) % 2 == 0 or (ni + nj) % 2 == 0:
            return 0.0
        mi_f, ni_f = float(mi), float(ni)
        mj_f, nj_f = float(mj), float(nj)
        Ix = 2.0 * a * mi_f / (np.pi * (mi_f**2 - mj_f**2))
        Iy = 2.0 * b * ni_f / (np.pi * (ni_f**2 - nj_f**2))
        K_ij = -2.0 * np.pi**4 * Ix * Iy * mj_f * nj_f * (
            D16 * (mi_f**2 + mj_f**2) / (a**3 * b)
            + D26 * (ni_f**2 + nj_f**2) / (a * b**3)
        )
        G_ii = (a * b / 4.0) * np.pi**2 * (mi_f / a)**2
        return K_ij / G_ii

    # Start with the uncoupled (orthotropic) minimum across all modes
    candidates = [_H_diag(m, n) for m, n in modes]

    # For each pair of modes that couple, compute the 2x2 closed-form eigenvalue.
    # lambda_min = (a+b)/2 - sqrt((a-b)^2/4 + c^2)
    for i, (mi, ni) in enumerate(modes):
        for j, (mj, nj) in enumerate(modes):
            if j <= i:
                continue
            if (mi + mj) % 2 == 0 or (ni + nj) % 2 == 0:
                continue

            h_ii = _H_diag(mi, ni)
            h_jj = _H_diag(mj, nj)
            h_ij = _H_offdiag(mi, ni, mj, nj)

            # Note: H is NOT symmetric when divided by G_ii (different G for each row).
            # For a proper 2x2 eigenvalue, symmetrise via: H_sym = G^{-1/2} K G^{-1/2}.
            # But the generalised eigenvalue K*v = lambda*G*v for 2x2 with diagonal G
            # has eigenvalues from det(K - lambda*G) = 0:
            #   (K11 - lam*G1)(K22 - lam*G2) - K12^2 = 0
            #   G1*G2*lam^2 - (K11*G2 + K22*G1)*lam + (K11*K22 - K12^2) = 0
            mi_f, ni_f = float(mi), float(ni)
            mj_f, nj_f = float(mj), float(nj)
            G_i = (a * b / 4.0) * np.pi**2 * (mi_f / a)**2
            G_j = (a * b / 4.0) * np.pi**2 * (mj_f / a)**2
            K_ii = h_ii * G_i
            K_jj = h_jj * G_j
            K_ij_val = h_ij * G_i  # K_ij (actual, not divided by G)

            # Quadratic: G_i*G_j*lam^2 - (K_ii*G_j + K_jj*G_i)*lam + (K_ii*K_jj - K_ij^2) = 0
            A_coeff = G_i * G_j
            B_coeff = -(K_ii * G_j + K_jj * G_i)
            C_coeff = K_ii * K_jj - K_ij_val**2

            disc = B_coeff**2 - 4.0 * A_coeff * C_coeff
            lam_min = (-B_coeff - np.sqrt(disc + eps)) / (2.0 * A_coeff + eps)

            candidates.append(lam_min)

    # Take minimum across all candidates
    Ncr = candidates[0]
    for k in range(1, len(candidates)):
        Ncr = np.fmin(Ncr, candidates[k])

    return Ncr


def _rr_Nyy_cr_smooth(D, a: float, b: float,
                       m_modes: tuple = (1, 2, 3),
                       n_modes: tuple = (1, 2),
                       eps: float = 1e-30):
    """CasADi-compatible Rayleigh-Ritz Nyy_cr — swap axes.
    Nemeth (2003), applied with coordinate swap."""
    D11, D12, D22, D66 = D[0, 0], D[0, 1], D[1, 1], D[2, 2]
    D16, D26           = D[0, 2], D[1, 2]
    # Swap axes: D_swap(i,j) maps xx<->yy for the transposed coordinate system.
    # Build element-by-element so it works for both numpy and CasADi MX.
    D_swap = D * 0.0          # preserve type (MX or ndarray), zero-fill
    D_swap[0, 0] = D22;  D_swap[0, 1] = D12;  D_swap[0, 2] = D26
    D_swap[1, 0] = D12;  D_swap[1, 1] = D11;  D_swap[1, 2] = D16
    D_swap[2, 0] = D26;  D_swap[2, 1] = D16;  D_swap[2, 2] = D66
    return _rr_Nxx_cr_smooth(D_swap, b, a, m_modes, n_modes, eps)


def buckling_rf_smooth(Nxx, Nyy, Nxy, D, a: float, b: float,
                       m_x: int = 1, m_y: int = 1, eps: float = 1e-30):
    """
    CasADi-compatible combined buckling RF with Rayleigh-Ritz Ncr.

    Uses (m,n) mode Rayleigh-Ritz basis that properly accounts for D16/D26
    bend-twist coupling via Gershgorin lower bound on the eigenvalue.
    Nemeth (2003), NASA/TP-2003-212131; Whitney (1987) interaction.

    m_x, m_y select which half-wave modes to centre the Ritz basis on.
    """
    # Build mode tuples centred on the suggested mode numbers
    m_modes_x = tuple(sorted(set(max(m, 1) for m in (m_x - 1, m_x, m_x + 1))))
    m_modes_y = tuple(sorted(set(max(m, 1) for m in (m_y - 1, m_y, m_y + 1))))
    while len(m_modes_x) < 3:
        m_modes_x = m_modes_x + (m_modes_x[-1] + 1,)
    while len(m_modes_y) < 3:
        m_modes_y = m_modes_y + (m_modes_y[-1] + 1,)

    n_modes = (1, 2)  # transverse modes needed for D16/D26 coupling

    Ncx  = _rr_Nxx_cr_smooth(D, a, b, m_modes_x, n_modes)
    Ncy  = _rr_Nyy_cr_smooth(D, a, b, m_modes_y, n_modes)
    Ncxy = Nxy_cr_smooth(D, a, b, eps)

    Rx  = np.fmax(-Nxx, 0.0) / (Ncx  + eps)
    Ry  = np.fmax(-Nyy, 0.0) / (Ncy  + eps)
    Rs  = np.sqrt(Nxy**2 + eps) / (Ncxy + eps)

    lin  = Rx + Ry
    disc = lin**2 + 4.0 * Rs**2
    RF   = (-lin + np.sqrt(disc + eps)) / (2.0 * Rs**2 + eps)
    return RF


def suggest_mode_number(a: float, b: float, D: _np.ndarray) -> int:
    """Optimal half-wave mode m* that minimises Nxx_cr (evaluates m=1..10)."""
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
    t_ply = 0.125e-3

    angles = [0, 45, -45, 90, 90, -45, 45, 0]
    plies  = [Ply(mat, t_ply, a) for a in angles]
    lam    = Laminate(plies)
    D      = lam.D

    a, b = 0.5, 0.15

    print(f"Panel {a:.2f} m x {b:.2f} m,  h = {lam.thickness*1e3:.2f} mm")
    print(f"  Nxx_cr = {Nxx_cr(D, a, b)/1e3:.1f} kN/m")
    print(f"  Nyy_cr = {Nyy_cr(D, a, b)/1e3:.1f} kN/m")
    print(f"  Nxy_cr = {Nxy_cr(D, a, b)/1e3:.1f} kN/m")
    print(f"  Suggested mode m* = {suggest_mode_number(a, b, D)}")
    print()

    N_applied = _np.array([-280e3, -115e3, 42e3])
    rf = buckling_rf(N_applied, D, a, b)
    print(f"Applied: Nxx={N_applied[0]/1e3:.0f}, Nyy={N_applied[1]/1e3:.0f}, "
          f"Nxy={N_applied[2]/1e3:.0f} kN/m")
    print(f"Buckling RF = {rf:.3f}  ({'OK' if rf >= 1.0 else 'BUCKLED'})")
