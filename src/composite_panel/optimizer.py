"""
composite_panel.optimizer
-------------------------
Minimum-mass symmetric laminate via IPOPT/CasADi (AeroSandbox Opti).

    min  rho * 2 * sum(t_k)
    s.t. Tsai-Wu RF >= rf_min  for each ply
         t_k >= t_min
         balance: t_i == t_j,  theta_j == -theta_i

Q_bar invariant form (Kassapoglou 2013 §2.4) — pure trig, no T^-1,
differentiable wrt angles.  B=0 enforced by construction; two 3x3 solves.
  A = 2*sum(Q_bar_k*t_k),  D = 2*sum(Q_bar_k*(z1^3-z0^3)/3)
Tsai-Wu RF: a*RF^2 + b*RF - 1 = 0  →  RF = (-b + sqrt(b^2+4a))/(2a)

Refs: Kassapoglou (2013), Tsai & Wu J. Composite Materials 5(1) 1971
"""

import aerosandbox as asb
import aerosandbox.numpy as np      # CasADi-compatible drop-in
import numpy as _np                 # standard numpy for constants
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

from .ply import PlyMaterial, Ply, IM7_8552
from .laminate import Laminate
from .failure import check_laminate
from .aero_loads import (
    supersonic_panel_loads, WingGeometry, wing_panel_loads,
)
from .loads_db import LoadCase, LoadsDatabase
from .thermal import (
    PlyThermal, ThermalState, IM7_8552_thermal,
    alpha_bar as _alpha_bar, thermal_resultants as _thermal_resultants,
)
from .buckling import (
    buckling_rf_smooth as _buckling_rf_smooth,
    buckling_rf        as _buckling_rf,
    Nxx_cr_smooth, suggest_mode_number,
)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class OptimizationResult:
    """Outcome of a single laminate optimisation run."""
    angles_half:    List[float]    # half-stack angles [deg]  (optimised if optimize_angles=True)
    t_half:         _np.ndarray    # optimised half-stack thicknesses [m]
    t_full:         _np.ndarray    # full symmetric stack [m]
    total_h:        float          # total laminate thickness [m]
    areal_density:  float          # rho * h  [kg/m2]
    min_tsai_wu_rf: float          # governing Tsai-Wu RF at optimum
    rf_min_target:  float          # rf_min constraint used
    converged:      bool

    def summary(self) -> str:
        full_ang = list(self.angles_half) + list(reversed(self.angles_half))
        stack_str = "/".join(f"{a:.1f}" for a in full_ang)
        lines = [
            "Optimisation result",
            f"  Stack   : [{stack_str}]",
            f"  t_half  : {[f'{t*1e3:.3f}mm' for t in self.t_half]}",
            f"  h total : {self.total_h*1e3:.3f} mm",
            f"  rho*h   : {self.areal_density:.4f} kg/m2",
            f"  min RF  : {self.min_tsai_wu_rf:.4f}  (target >= {self.rf_min_target:.2f})",
            f"  converged: {self.converged}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Analytical Q_bar — symbolic-compatible (no T^-1)
# ---------------------------------------------------------------------------

def _Q_bar_matrix(mat: PlyMaterial, angle_rad):
    """
    3x3 rotated stiffness matrix using the invariant polynomial form.

    Works for both float and CasADi symbolic inputs:
      - float input  -> standard numpy (3,3) array  (precomputed constant)
      - CasADi input -> (3,3) CasADi MX expression  (inside opti graph)

    Invariant form (Kassapoglou 2013 eq 2.33):
      Q_bar_11 = Q11*c^4 + 2*(Q12+2*Q66)*c^2*s^2 + Q22*s^4
      Q_bar_22 = Q11*s^4 + 2*(Q12+2*Q66)*c^2*s^2 + Q22*c^4
      Q_bar_12 = (Q11+Q22-4*Q66)*c^2*s^2 + Q12*(c^4+s^4)
      Q_bar_66 = (Q11+Q22-2*Q12-2*Q66)*c^2*s^2 + Q66*(c^4+s^4)
      Q_bar_16 = (Q11-Q12-2*Q66)*c^3*s - (Q22-Q12-2*Q66)*c*s^3
      Q_bar_26 = (Q11-Q12-2*Q66)*c*s^3 - (Q22-Q12-2*Q66)*c^3*s
    """
    denom = 1.0 - mat.nu12 * mat.nu21
    Q11 = mat.E1 / denom
    Q22 = mat.E2 / denom
    Q12 = mat.nu12 * mat.E2 / denom
    Q66 = mat.G12

    c = np.cos(angle_rad)   # asb.numpy: float -> float, CasADi -> MX
    s = np.sin(angle_rad)
    c2, s2 = c*c, s*s
    c4, s4 = c2*c2, s2*s2
    c2s2   = c2*s2
    c3s    = c2*c*s
    cs3    = c*s2*s

    Qb11 = Q11*c4 + 2*(Q12+2*Q66)*c2s2 + Q22*s4
    Qb22 = Q11*s4 + 2*(Q12+2*Q66)*c2s2 + Q22*c4
    Qb12 = (Q11+Q22-4*Q66)*c2s2 + Q12*(c4+s4)
    Qb66 = (Q11+Q22-2*Q12-2*Q66)*c2s2 + Q66*(c4+s4)
    Qb16 = (Q11-Q12-2*Q66)*c3s - (Q22-Q12-2*Q66)*cs3
    Qb26 = (Q11-Q12-2*Q66)*cs3 - (Q22-Q12-2*Q66)*c3s

    return np.array([
        [Qb11, Qb12, Qb16],
        [Qb12, Qb22, Qb26],
        [Qb16, Qb26, Qb66],
    ])


def _T_stress(angle_rad):
    """
    3x3 stress transformation matrix.  CasADi-compatible.
    """
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array([
        [ c*c,   s*s,   2*c*s ],
        [ s*s,   c*c,  -2*c*s ],
        [-c*s,   c*s,  c*c-s*s],
    ])


# ---------------------------------------------------------------------------
# Differentiable ABD assembly for symmetric laminates
# ---------------------------------------------------------------------------

def _build_ABD_symmetric(t_half: list, Q_bars_half: list):
    """
    Assemble A and D for a symmetric laminate.  CasADi-compatible.

    For symmetric half-stack [t0..t_{n-1}]:
      A = 2 * sum_k  Q_bar_k * t_k                        (factor-of-2 from top mirror)
      D = 2 * sum_k  Q_bar_k * (z1_k^3 - z0_k^3) / 3
    where z interfaces are computed for the bottom half only.

    Q_bars_half may contain numpy arrays (fixed angles) or CasADi expressions
    (variable angles) — arithmetic is the same either way.
    """
    n      = len(t_half)
    h_half = sum(t_half)           # CasADi scalar if t_half contains opti vars

    # A — linear in t
    A = Q_bars_half[0] * (2.0 * t_half[0])
    for k in range(1, n):
        A = A + Q_bars_half[k] * (2.0 * t_half[k])

    # D — cubic in t via z-coordinates; accumulate z from bottom face
    z0 = -h_half
    z1 = z0 + t_half[0]
    D  = Q_bars_half[0] * (2.0 * (z1**3 - z0**3) / 3.0)
    z0 = z1
    for k in range(1, n):
        z1 = z0 + t_half[k]
        D  = D + Q_bars_half[k] * (2.0 * (z1**3 - z0**3) / 3.0)
        z0 = z1

    return A, D


def _ply_zmid_symmetric(t_half: list) -> list:
    """
    Mid-plane z for all 2n plies of a symmetric stack.
    Full order: [ply0..ply_{n-1}, ply_{n-1}..ply_0] (bottom -> top).
    Top-half z-mids are negatives of the corresponding bottom values.
    Returns list of 2n CasADi scalars.
    """
    n      = len(t_half)
    h_half = sum(t_half)

    z_bot = []
    z = -h_half
    for k in range(n):
        z_bot.append(z + t_half[k] / 2.0)
        z = z + t_half[k]

    # Mirror: top ply j (j=0..n-1) corresponds to bottom ply (n-1-j)
    z_top = [-z_bot[n-1-j] for j in range(n)]

    return z_bot + z_top   # length 2n


# ---------------------------------------------------------------------------
# Smooth Tsai-Wu RF (branch-free)
# ---------------------------------------------------------------------------

def _tsai_wu_rf(s1, s2, s12, mat: PlyMaterial, eps: float = 1e-30):
    """
    Tsai-Wu RF via the quadratic root formula.  No if/else on stress sign.

      a*RF^2 + b*RF - 1 = 0
      RF = (-b + sqrt(b^2 + 4a)) / (2a)

    eps prevents 0/0 at zero-stress states; approaches 1/b smoothly as a->0.
    """
    F1  = 1.0/mat.F1t - 1.0/mat.F1c
    F2  = 1.0/mat.F2t - 1.0/mat.F2c
    F11 = 1.0/(mat.F1t * mat.F1c)
    F22 = 1.0/(mat.F2t * mat.F2c)
    F66 = 1.0/(mat.F12**2)
    F12i = -0.5 / np.sqrt(mat.F1t * mat.F1c * mat.F2t * mat.F2c)

    a = F11*s1**2 + F22*s2**2 + F66*s12**2 + 2.0*F12i*s1*s2
    b = F1*s1 + F2*s2

    return (-b + np.sqrt(b**2 + 4.0*a + eps)) / (2.0*a + eps)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def detect_balance_pairs(angles_deg: List[float], tol: float = 1.0) -> List[Tuple[int,int]]:
    """
    Auto-detect +theta/-theta index pairs in a half-stack angle list.

    Skips angles near 0 or 90 (they are self-paired by the symmetric layup
    and do not need an explicit balance constraint).

    Example
    -------
    detect_balance_pairs([0, 45, -45, 90])   -> [(1, 2)]
    detect_balance_pairs([0, 30, -30, 60, -60, 90]) -> [(1, 2), (3, 4)]
    """
    pairs, used = [], set()
    for i, ai in enumerate(angles_deg):
        if i in used or abs(ai) < tol or abs(abs(ai) - 90) < tol:
            continue
        for j in range(i+1, len(angles_deg)):
            if j in used:
                continue
            if abs(angles_deg[j] + ai) < tol:
                pairs.append((i, j))
                used.add(i); used.add(j)
                break
    return pairs


# ---------------------------------------------------------------------------
# Core optimiser
# ---------------------------------------------------------------------------

def optimize_laminate(
    N_loads:          _np.ndarray,
    M_loads:          _np.ndarray,
    mat:              PlyMaterial,
    angles_half_deg:  List[float],
    rho_kg_m3:        float = 1600.0,
    rf_min:           float = 1.5,
    t_min:            float = 0.05e-3,
    t_init:           float = 0.125e-3,
    balance_pairs:    Optional[List[Tuple[int,int]]] = None,
    optimize_angles:  bool  = False,
    angle_bounds_deg: Optional[List[Tuple[float,float]]] = None,
    # -- thermal -----------------------------------------------------------------
    thermal_state:    Optional[ThermalState]  = None,
    ply_thermal:      Optional[PlyThermal]    = None,
    # -- buckling ----------------------------------------------------------------
    panel_a:          Optional[float] = None,   # spanwise panel dimension [m]
    panel_b:          Optional[float] = None,   # chordwise panel dimension [m]
    buckle_rf_min:    float = 1.0,              # buckling RF requirement
    # ----------------------------------------------------------------------------
    verbose:          bool  = True,
) -> OptimizationResult:
    """
    Minimum-mass laminate optimiser.

    Parameters
    ----------
    N_loads          : [Nxx, Nyy, Nxy]  [N/m]
    M_loads          : [Mxx, Myy, Mxy]  [N.m/m]
    mat              : PlyMaterial (elastic constants + Tsai-Wu allowables)
    angles_half_deg  : half-stack fibre angles [deg] — initial guesses (or fixed values)
    rho_kg_m3        : cured ply density [kg/m3]
    rf_min           : minimum required Tsai-Wu RF
    t_min            : minimum ply thickness [m]
    t_init           : initial guess for each ply thickness [m]
    balance_pairs    : list of (i, j) pairs to enforce t_i = t_j.
                       When optimize_angles=True, also enforces theta_j = -theta_i.
    optimize_angles  : if True, fibre angles become CasADi design variables
    angle_bounds_deg : per-ply angle bounds [(lo, hi), ...].
                       Use (x, x) to fix a ply at angle x degrees.
                       Defaults to (-90, 90) for all plies if None.
    verbose          : pass through to IPOPT output

    Returns
    -------
    OptimizationResult
    """
    n_half = len(angles_half_deg)
    N_vec  = np.array(N_loads, dtype=float)   # asb.numpy: keeps in CasADi graph if symbolic
    M_vec  = np.array(M_loads, dtype=float)

    # -- Thermal load resultants (added to mechanical loads) -----------------
    # N_T and M_T are computed in standard numpy from fixed angles + thermal state.
    # They are constants inside the opti problem (loads don't depend on t_k).
    if thermal_state is not None:
        pt = ply_thermal if ply_thermal is not None else IM7_8552_thermal()
        # Build a temporary laminate at initial-guess thicknesses for z_interfaces
        _ang_full_init = [_np.degrees(r) for r in
                          [_np.radians(a) for a in angles_half_deg] +
                          list(reversed([_np.radians(a) for a in angles_half_deg]))]
        _lam_init = Laminate([Ply(mat, t_init, a) for a in _ang_full_init])
        _N_T, _M_T = _thermal_resultants(
            _lam_init.plies,
            [pt] * (2 * n_half),
            thermal_state,
            _lam_init.z_interfaces,
        )
        N_vec = N_vec + _N_T
        M_vec = M_vec + _M_T

    # Default angle bounds
    if angle_bounds_deg is None:
        angle_bounds_deg = [(-90.0, 90.0)] * n_half

    # -----------------------------------------------------------------------
    opti = asb.Opti()

    # -- Thickness design variables -----------------------------------------
    t = opti.variable(n_vars=n_half, init_guess=t_init, lower_bound=t_min)
    t_list = [t[k] for k in range(n_half)]

    # -- Angle design variables (optional) ----------------------------------
    if optimize_angles:
        theta_init = _np.radians([a for a in angles_half_deg])
        lo = _np.radians([b[0] for b in angle_bounds_deg])
        hi = _np.radians([b[1] for b in angle_bounds_deg])
        theta = opti.variable(n_vars=n_half, init_guess=theta_init,
                              lower_bound=lo, upper_bound=hi)
        theta_list = [theta[k] for k in range(n_half)]

        # Balance: theta_j = -theta_i
        if balance_pairs is not None:
            for (i, j) in balance_pairs:
                opti.subject_to(theta_list[j] == -theta_list[i])
    else:
        theta_list = [_np.radians(a) for a in angles_half_deg]

    # Balance: t_i = t_j
    if balance_pairs is not None:
        for (i, j) in balance_pairs:
            opti.subject_to(t_list[i] == t_list[j])

    # -- Q_bar matrices (CasADi if angles are variables) --------------------
    Q_bars_half = [_Q_bar_matrix(mat, theta_list[k]) for k in range(n_half)]

    # Full-stack sequences (bottom -> top)
    theta_full  = theta_list + list(reversed(theta_list))
    Q_bars_full = Q_bars_half + list(reversed(Q_bars_half))

    # -- ABD assembly -------------------------------------------------------
    A_mat, D_mat = _build_ABD_symmetric(t_list, Q_bars_half)

    # -- CLT response (B=0 for symmetric layup) -----------------------------
    a_comp = np.linalg.inv(A_mat)
    d_comp = np.linalg.inv(D_mat)
    eps0   = a_comp @ N_vec
    kappa  = d_comp @ M_vec

    # -- Ply stresses + Tsai-Wu constraints ---------------------------------
    z_mids = _ply_zmid_symmetric(t_list)

    for k in range(2 * n_half):
        Qb    = Q_bars_full[k]
        T_k   = _T_stress(theta_full[k])   # CasADi if optimize_angles
        z_k   = z_mids[k]

        eps_xy = eps0 + z_k * kappa
        sig_xy = Qb  @ eps_xy
        sig_12 = T_k @ sig_xy

        rf_k = _tsai_wu_rf(sig_12[0], sig_12[1], sig_12[2], mat)
        opti.subject_to(rf_k >= rf_min)

    # -- Buckling constraint (differentiable through D_mat) ------------------
    if panel_a is not None and panel_b is not None:
        m_x = max(1, round(panel_a / panel_b))
        RF_buckle = _buckling_rf_smooth(
            N_vec[0], N_vec[1], N_vec[2],
            D_mat, panel_a, panel_b, m_x, 1,
        )
        opti.subject_to(RF_buckle >= buckle_rf_min)

    # -- Objective ----------------------------------------------------------
    opti.minimize(rho_kg_m3 * 2.0 * sum(t_list))

    # -- Solve --------------------------------------------------------------
    try:
        sol = opti.solve(verbose=verbose)
        converged = True
        t_half_opt = _np.array([float(sol(t[k])) for k in range(n_half)])
        if optimize_angles:
            ang_half_opt = [float(_np.degrees(sol(theta[k]))) for k in range(n_half)]
        else:
            ang_half_opt = list(angles_half_deg)
    except Exception as exc:
        if verbose:
            print(f"  Optimizer did not converge: {exc}")
        converged    = False
        t_half_opt   = _np.full(n_half, t_init)
        ang_half_opt = list(angles_half_deg)

    # -- Post-process with standard numpy CLT -------------------------------
    t_full_opt  = _np.concatenate([t_half_opt, t_half_opt[::-1]])
    ang_full    = ang_half_opt + list(reversed(ang_half_opt))
    h_opt       = float(t_full_opt.sum())

    lam_chk = Laminate([Ply(mat, t_full_opt[k], ang_full[k]) for k in range(2*n_half)])
    resp    = lam_chk.response(N=N_vec, M=M_vec)
    fails   = check_laminate(resp, lam_chk.plies, criterion="tsai_wu", verbose=False)
    min_rf  = float(min(r.rf for r in fails))

    return OptimizationResult(
        angles_half    = ang_half_opt,
        t_half         = t_half_opt,
        t_full         = t_full_opt,
        total_h        = h_opt,
        areal_density  = rho_kg_m3 * h_opt,
        min_tsai_wu_rf = min_rf,
        rf_min_target  = rf_min,
        converged      = converged,
    )


# ---------------------------------------------------------------------------
# Multi-load-case envelope optimizer
# ---------------------------------------------------------------------------

@dataclass
class MulticaseOptimizationResult:
    """
    Outcome of a multi-load-case laminate optimisation.

    The key addition over OptimizationResult is governing_case — which load
    case is critical at the optimum for each ply.  This tells the designer
    which flight condition is driving the structural weight.

    Attributes
    ----------
    base            : OptimizationResult  (thicknesses, areal density, etc.)
    n_cases         : number of load cases considered simultaneously
    governing_cases : list of load case names, one per ply (bottom → top)
                      — the case with lowest Tsai-Wu RF at optimum
    rf_per_case     : dict mapping case name → min RF across all plies
                      (useful for identifying near-critical conditions)
    """
    base:             OptimizationResult
    n_cases:          int
    governing_cases:  List[str]     # length = n_full_plies
    rf_per_case:      dict          # {case_name: min_RF}

    # Delegate common attributes to the wrapped result for convenience
    @property
    def t_full(self):         return self.base.t_full
    @property
    def t_half(self):         return self.base.t_half
    @property
    def total_h(self):        return self.base.total_h
    @property
    def areal_density(self):  return self.base.areal_density
    @property
    def min_tsai_wu_rf(self): return self.base.min_tsai_wu_rf
    @property
    def converged(self):      return self.base.converged

    def summary(self) -> str:
        lines = [self.base.summary(),
                 f"  n_cases : {self.n_cases}",
                 "  Governing case per ply:"]
        for k, name in enumerate(self.governing_cases):
            lines.append(f"    ply {k:2d} → {name}")
        lines.append("  RF per case:")
        for name, rf in sorted(self.rf_per_case.items(), key=lambda x: x[1]):
            lines.append(f"    {name:<28}  RF = {rf:.4f}")
        return "\n".join(lines)


def optimize_laminate_multicase(
    load_cases:       "List[LoadCase] | LoadsDatabase",
    mat:              PlyMaterial,
    angles_half_deg:  List[float],
    rho_kg_m3:        float = 1600.0,
    rf_min:           float = 1.5,
    t_min:            float = 0.05e-3,
    t_init:           float = 0.125e-3,
    balance_pairs:    Optional[List[Tuple[int,int]]] = None,
    panel_a:          Optional[float] = None,
    panel_b:          Optional[float] = None,
    buckle_rf_min:    float = 1.0,
    verbose:          bool  = True,
) -> MulticaseOptimizationResult:
    """
    Minimum-mass laminate sized simultaneously for ALL provided load cases.

    MOTIVATION
    ----------
    In real structural sizing, the laminate must survive every load case in
    the flight envelope simultaneously — not just the worst individual case.
    Two conditions that are each separately safe may together require a
    heavier laminate because they load different plies in different ways.

    For example: a high-Mach case drives Nyy (chordwise) and requires 90°
    plies; a high-g maneuver drives Nxx (spanwise) and requires 0° plies.
    The single-case optimizer run at the "worst" condition misses the
    interaction — the multi-case formulation captures it.

    FORMULATION
    -----------
    The NLP is identical to optimize_laminate() with one extension:
    the Tsai-Wu constraints are repeated for EVERY load case:

        for each case c in load_cases:
            eps0_c = A^-1 * N_c
            kappa_c = D^-1 * M_c
            for each ply k:
                RF_k(c) = tsai_wu(sigma_12_k(eps0_c, kappa_c, z_k)) >= rf_min

    All constraints share the same decision variables (ply thicknesses t_k),
    so IPOPT finds the lightest laminate that is simultaneously safe under
    every condition.  The number of constraints is n_plies × n_cases.

    DESIGN VARIABLE COUNT
    ---------------------
    Same as optimize_laminate(): n_half ply thicknesses.
    CONSTRAINT COUNT: 2 * n_half * n_cases (Tsai-Wu) + n_half (t >= t_min)
                    + optional buckling constraints per case

    Parameters
    ----------
    load_cases      : list of LoadCase objects or a LoadsDatabase
    mat             : PlyMaterial
    angles_half_deg : half-stack fibre angles [deg]
    rho_kg_m3       : cured ply density [kg/m³]
    rf_min          : Tsai-Wu RF requirement (applies to ALL cases)
    t_min           : minimum ply gauge [m]
    t_init          : initial thickness guess [m]
    balance_pairs   : +/-theta thickness equality pairs
    panel_a, panel_b: panel dimensions [m] for buckling — if given, buckling
                      constraint RF >= buckle_rf_min is added for each case
    buckle_rf_min   : combined buckling RF requirement
    verbose         : IPOPT console output

    Returns
    -------
    MulticaseOptimizationResult
    """
    # Allow either a list of LoadCase or a LoadsDatabase
    if isinstance(load_cases, LoadsDatabase):
        cases = load_cases.cases
    else:
        cases = list(load_cases)

    if not cases:
        raise ValueError("load_cases must contain at least one LoadCase")

    n_half = len(angles_half_deg)
    n_full = 2 * n_half

    opti = asb.Opti()

    # ── Design variables: ply thicknesses ────────────────────────────────────
    t      = opti.variable(n_vars=n_half, init_guess=t_init, lower_bound=t_min)
    t_list = [t[k] for k in range(n_half)]

    # ── Fixed angles (angle optimization not implemented for multi-case) ──────
    theta_list  = [_np.radians(a) for a in angles_half_deg]
    Q_bars_half = [_Q_bar_matrix(mat, theta_list[k]) for k in range(n_half)]
    theta_full  = theta_list + list(reversed(theta_list))
    Q_bars_full = Q_bars_half + list(reversed(Q_bars_half))

    # ── Balance: t_i == t_j ───────────────────────────────────────────────────
    if balance_pairs is not None:
        for (i, j) in balance_pairs:
            opti.subject_to(t_list[i] == t_list[j])

    # ── ABD assembly — shared across all load cases ───────────────────────────
    A_mat, D_mat = _build_ABD_symmetric(t_list, Q_bars_half)
    a_comp = np.linalg.inv(A_mat)
    d_comp = np.linalg.inv(D_mat)
    z_mids = _ply_zmid_symmetric(t_list)

    # ── Tsai-Wu constraints for EVERY load case ───────────────────────────────
    for case in cases:
        N_c = np.array(case.N, dtype=float)   # asb.numpy: preserves differentiability
        M_c = np.array(case.M, dtype=float)

        eps0  = a_comp @ N_c
        kappa = d_comp @ M_c

        for k in range(n_full):
            Qb    = Q_bars_full[k]
            T_k   = _T_stress(theta_full[k])
            z_k   = z_mids[k]

            eps_xy = eps0 + z_k * kappa
            sig_xy = Qb   @ eps_xy
            sig_12 = T_k  @ sig_xy

            rf_k = _tsai_wu_rf(sig_12[0], sig_12[1], sig_12[2], mat)
            opti.subject_to(rf_k >= rf_min)

        # ── Optional buckling constraint per case ─────────────────────────────
        if panel_a is not None and panel_b is not None:
            m_x = max(1, round(panel_a / panel_b))
            RF_b = _buckling_rf_smooth(
                N_c[0], N_c[1], N_c[2],
                D_mat, panel_a, panel_b, m_x, 1,
            )
            opti.subject_to(RF_b >= buckle_rf_min)

    # ── Objective: minimum areal mass ─────────────────────────────────────────
    opti.minimize(rho_kg_m3 * 2.0 * sum(t_list))

    # ── Solve ─────────────────────────────────────────────────────────────────
    try:
        sol       = opti.solve(verbose=verbose)
        converged = True
        t_half_opt = _np.array([float(sol(t[k])) for k in range(n_half)])
    except Exception as exc:
        if verbose:
            print(f"  Optimizer did not converge: {exc}")
        converged  = False
        t_half_opt = _np.full(n_half, t_init)

    ang_half_opt = list(angles_half_deg)

    # ── Post-process: standard numpy to find governing case per ply ──────────
    t_full_opt = _np.concatenate([t_half_opt, t_half_opt[::-1]])
    ang_full   = ang_half_opt + list(reversed(ang_half_opt))
    h_opt      = float(t_full_opt.sum())

    lam_chk = Laminate([Ply(mat, t_full_opt[k], ang_full[k]) for k in range(n_full)])

    # Track min RF per ply per case — find which case governs each ply
    ply_min_rf   = _np.full(n_full, _np.inf)   # running minimum RF per ply
    governing    = [""] * n_full               # case name that gives that min
    rf_per_case  = {}                           # {case_name: min_RF_across_all_plies}

    for case in cases:
        resp  = lam_chk.response(N=case.N, M=case.M)
        fails = check_laminate(resp, lam_chk.plies, criterion="tsai_wu",
                               verbose=False)
        case_min = _np.inf
        for k, r in enumerate(fails):
            case_min = min(case_min, r.rf)
            if r.rf < ply_min_rf[k]:
                ply_min_rf[k]  = r.rf
                governing[k]   = case.name
        rf_per_case[case.name] = float(case_min)

    overall_min_rf = float(ply_min_rf.min())

    base = OptimizationResult(
        angles_half    = ang_half_opt,
        t_half         = t_half_opt,
        t_full         = t_full_opt,
        total_h        = h_opt,
        areal_density  = rho_kg_m3 * h_opt,
        min_tsai_wu_rf = overall_min_rf,
        rf_min_target  = rf_min,
        converged      = converged,
    )

    return MulticaseOptimizationResult(
        base            = base,
        n_cases         = len(cases),
        governing_cases = governing,
        rf_per_case     = rf_per_case,
    )


# ---------------------------------------------------------------------------
# Pareto sweep
# ---------------------------------------------------------------------------

def pareto_sweep(
    N_loads:         _np.ndarray,
    M_loads:         _np.ndarray,
    mat:             PlyMaterial,
    angles_half_deg: List[float],
    rf_range:        Optional[_np.ndarray] = None,
    balance_pairs:   Optional[List[Tuple[int,int]]] = None,
    optimize_angles: bool = False,
    angle_bounds_deg: Optional[List[Tuple[float,float]]] = None,
    **kwargs,
) -> List[OptimizationResult]:
    """
    Sweep rf_min across rf_range and return one OptimizationResult per point.
    Builds the mass-vs-margin Pareto frontier for a given layup family.
    """
    if rf_range is None:
        rf_range = _np.linspace(1.0, 2.5, 16)

    results = []
    for rf in rf_range:
        r = optimize_laminate(
            N_loads, M_loads, mat, angles_half_deg,
            rf_min          = float(rf),
            balance_pairs   = balance_pairs,
            optimize_angles = optimize_angles,
            angle_bounds_deg= angle_bounds_deg,
            verbose         = False,
            **kwargs,
        )
        results.append(r)
    return results


# ---------------------------------------------------------------------------
# Multi-panel wing optimisation
# ---------------------------------------------------------------------------

@dataclass
class WingOptimizationResult:
    """Spanwise laminate sizing results across a wing half."""
    etas:        _np.ndarray        # spanwise stations (dimensionless)
    loads:       List               # PanelLoads at each station
    opt_results: List               # OptimizationResult at each station
    wing:        WingGeometry

    # -- convenience arrays -------------------------------------------------

    @property
    def Nxx(self) -> _np.ndarray:
        return _np.array([l.Nxx for l in self.loads])

    @property
    def Nyy(self) -> _np.ndarray:
        return _np.array([l.Nyy for l in self.loads])

    @property
    def Nxy(self) -> _np.ndarray:
        return _np.array([l.Nxy for l in self.loads])

    @property
    def thicknesses(self) -> _np.ndarray:
        return _np.array([r.total_h for r in self.opt_results])

    @property
    def areal_densities(self) -> _np.ndarray:
        return _np.array([r.areal_density for r in self.opt_results])

    @property
    def min_rfs(self) -> _np.ndarray:
        return _np.array([r.min_tsai_wu_rf for r in self.opt_results])

    @property
    def t_half_matrix(self) -> _np.ndarray:
        """(n_stations, n_half) array of half-stack thicknesses."""
        return _np.array([r.t_half for r in self.opt_results])

    @property
    def total_skin_mass(self) -> float:
        """
        Approximate total upper-skin mass [kg] by integrating areal density
        along the semi-span, accounting for chord variation.
        """
        masses = _np.array([
            self.opt_results[i].areal_density * self.wing.chord(self.etas[i])
            for i in range(len(self.etas))
        ])
        return float(_np.trapezoid(masses, self.etas * self.wing.semi_span))


def optimize_wing(
    wing:             WingGeometry,
    mach:             float,
    altitude_m:       float,
    alpha_deg:        float,
    mat:              PlyMaterial,
    angles_half_deg:  List[float],
    n_load:           float = 2.5,
    n_stations:       int   = 10,
    rf_min:           float = 1.5,
    t_min:            float = 0.05e-3,
    t_init:           float = 0.125e-3,
    balance_pairs:    Optional[List[Tuple[int,int]]] = None,
    optimize_angles:  bool  = False,
    angle_bounds_deg: Optional[List[Tuple[float,float]]] = None,
    rho_kg_m3:        float = 1600.0,
    panel_a:          Optional[float] = None,
    panel_b:          Optional[float] = None,
    buckle_rf_min:    float = 1.0,
    thermal_states:   Optional[List] = None,
    ply_thermal:      Optional[object] = None,
) -> WingOptimizationResult:
    """
    Panel-by-panel laminate optimisation across the wing semi-span.

    Runs optimize_laminate() independently at n_stations spanwise stations
    eta in [0.05, 0.95].  Panels are not structurally coupled — this is a
    local sizing approach appropriate for preliminary design.

    Each panel gets its own load state from wing_panel_loads(), which
    combines Ackeret pressure (Nyy, Mxx) with elliptic bending (Nxx).
    Load magnitude decreases from root to tip: the optimizer returns
    thinner laminates near the tip, giving the characteristic spanwise
    taper of a composite wing skin.

    Parameters
    ----------
    wing             : WingGeometry (planform + MTOW)
    mach             : cruise Mach (> 1)
    altitude_m       : cruise altitude [m]
    alpha_deg        : cruise angle of attack [deg]
    mat              : PlyMaterial
    angles_half_deg  : half-stack angle sequence [deg]
    n_load           : ultimate load factor
    n_stations       : number of spanwise analysis stations
    rf_min           : Tsai-Wu RF requirement (all plies, all stations)
    t_min            : minimum ply thickness [m]
    t_init           : initial ply thickness guess [m]
    balance_pairs    : +/-theta thickness (and optionally angle) equality pairs
    optimize_angles  : if True, ply angles also become design variables
    angle_bounds_deg : per-ply angle bounds for optimize_angles=True
    rho_kg_m3        : cured ply density [kg/m3]

    Returns
    -------
    WingOptimizationResult
    """
    etas       = _np.linspace(0.05, 0.95, n_stations)
    loads_list = []
    opt_list   = []

    print(f"  Sizing {n_stations} spanwise stations — RF >= {rf_min}  t_min = {t_min*1e3:.2f} mm")

    for i, eta in enumerate(etas):
        panel_loads = wing_panel_loads(wing, eta, mach, altitude_m, alpha_deg, n_load)
        ts = thermal_states[i] if thermal_states is not None else None

        result = optimize_laminate(
            N_loads          = panel_loads.N,
            M_loads          = panel_loads.M,
            mat              = mat,
            angles_half_deg  = angles_half_deg,
            rf_min           = rf_min,
            t_min            = t_min,
            t_init           = t_init,
            balance_pairs    = balance_pairs,
            optimize_angles  = optimize_angles,
            angle_bounds_deg = angle_bounds_deg,
            rho_kg_m3        = rho_kg_m3,
            panel_a          = panel_a,
            panel_b          = panel_b,
            buckle_rf_min    = buckle_rf_min,
            thermal_state    = ts,
            ply_thermal      = ply_thermal,
            verbose          = False,
        )

        loads_list.append(panel_loads)
        opt_list.append(result)

        bar  = "#" * (i + 1) + "." * (n_stations - i - 1)
        conv = "OK" if result.converged else "!!"
        print(f"\r  [{bar}]  eta={eta:.2f}  h={result.total_h*1e3:.2f}mm"
              f"  RF={result.min_tsai_wu_rf:.3f}  [{conv}]", end="", flush=True)

    print()
    total_mass = _np.trapezoid(
        [opt_list[i].areal_density * wing.chord(etas[i]) for i in range(n_stations)],
        etas * wing.semi_span,
    )
    print(f"  Upper-skin mass (semi-span): {total_mass:.2f} kg")

    return WingOptimizationResult(
        etas=etas, loads=loads_list, opt_results=opt_list, wing=wing,
    )


# ---------------------------------------------------------------------------
# Demo — python optimize_laminate.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import matplotlib.cm as cm

    # ── Palette ─────────────────────────────────────────────────────────────
    BG    = "#0a0e1a"; BLUE  = "#00aaff"; GOLD  = "#f0a030"
    RED   = "#ff4455"; DIM   = "#3a4060"; WHITE = "#e8edf5"
    TEAL  = "#00ddbb"; GREEN = "#44dd88"

    def _style(ax):
        ax.set_facecolor(BG)
        ax.tick_params(colors=WHITE, labelsize=8)
        ax.xaxis.label.set_color(WHITE)
        ax.yaxis.label.set_color(WHITE)
        ax.title.set_color(WHITE)
        for spine in ax.spines.values():
            spine.set_edgecolor(DIM)
        ax.grid(color=DIM, linewidth=0.4, alpha=0.5)

    # ── Wing definition  (representative supersonic fighter class) ───────────
    wing = WingGeometry(
        semi_span    = 4.5,       # m  — ~9 m full span
        root_chord   = 4.0,       # m
        taper_ratio  = 0.25,      # cropped delta planform
        sweep_le_deg = 50.0,      # deg  — strongly swept leading edge
        t_over_c     = 0.04,      # 4% thin supersonic profile
        mtow_n       = 120_000.0, # N   — ~12 tonne class
    )

    # ── Flight condition ─────────────────────────────────────────────────────
    MACH      = 1.6
    ALT_M     = 15_000
    ALPHA_DEG = 4.0
    N_LOAD    = 2.5
    RHO_PLY   = 1600.0   # kg/m3  IM7/8552 cured density
    mat       = IM7_8552()

    # ── Layup families ───────────────────────────────────────────────────────
    # Family A: [0/+/-45/90]s  — standard aerospace quasi-isotropic
    FAM_A  = [0.0, 45.0, -45.0, 90.0]
    PAIRS_A = detect_balance_pairs(FAM_A)

    # Family B: [0/theta/-theta/90]s with continuous theta optimisation
    FAM_B_INIT  = [0.0, 45.0, -45.0, 90.0]
    PAIRS_B     = [(1, 2)]
    BOUNDS_B    = [(0.0, 0.0), (5.0, 85.0), (-85.0, -5.0), (90.0, 90.0)]

    print("=" * 65)
    print(f"  Wing skin sizing  |  Mach {MACH} @ {ALT_M/1e3:.0f} km  |  {mat.name}")
    print(f"  Planform: b/2={wing.semi_span}m  c_root={wing.root_chord}m"
          f"  lambda={wing.taper_ratio}  sweep={wing.sweep_le_deg}deg")
    print(f"  MTOW = {wing.mtow_n/1e3:.0f} kN  |  n = {N_LOAD}g  |  RF >= 1.5")
    print("=" * 65)

    # ── Wing sizing: Family A ────────────────────────────────────────────────
    print("\nFamily A  [0/+/-45/90]s — spanwise sizing ...")
    wing_A = optimize_wing(
        wing=wing, mach=MACH, altitude_m=ALT_M, alpha_deg=ALPHA_DEG,
        mat=mat, angles_half_deg=FAM_A,
        n_load=N_LOAD, n_stations=12, rf_min=1.5, t_min=0.05e-3, t_init=0.15e-3,
        balance_pairs=PAIRS_A, rho_kg_m3=RHO_PLY,
    )

    # ── Wing sizing: Family B (continuous theta) ─────────────────────────────
    print("\nFamily B  [0/theta/-theta/90]s continuous — spanwise sizing ...")
    wing_B = optimize_wing(
        wing=wing, mach=MACH, altitude_m=ALT_M, alpha_deg=ALPHA_DEG,
        mat=mat, angles_half_deg=FAM_B_INIT,
        n_load=N_LOAD, n_stations=12, rf_min=1.5, t_min=0.05e-3, t_init=0.15e-3,
        balance_pairs=PAIRS_B, optimize_angles=True, angle_bounds_deg=BOUNDS_B,
        rho_kg_m3=RHO_PLY,
    )

    # ── Figure: 4-panel wing skin map ────────────────────────────────────────
    fig = plt.figure(figsize=(15, 10), facecolor=BG)
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.44, wspace=0.30,
                            left=0.08, right=0.97, top=0.89, bottom=0.09)

    etas = wing_A.etas

    # ── (1) Spanwise load distribution ──────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(etas, wing_A.Nxx / 1e3, color=BLUE,  lw=2.0, label="Nxx (bending)")
    ax1.plot(etas, wing_A.Nyy / 1e3, color=RED,   lw=2.0, label="Nyy (pressure)")
    ax1.plot(etas, wing_A.Nxy / 1e3, color=GOLD,  lw=1.5, linestyle="--", label="Nxy (shear)")
    ax1.axhline(0, color=DIM, lw=0.5)
    ax1.set_xlabel("eta = y/b  [-]",  fontsize=9)
    ax1.set_ylabel("Running load  [kN/m]", fontsize=9)
    ax1.set_title("Spanwise load distribution", fontsize=10, pad=7)
    ax1.legend(fontsize=8, framealpha=0.15, labelcolor=WHITE)
    _style(ax1)

    # ── (2) Spanwise total thickness ─────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(etas, wing_A.thicknesses * 1e3, color=BLUE,
             lw=2.0, marker="o", markersize=4, label="[0/+/-45/90]s (discrete)")
    ax2.plot(etas, wing_B.thicknesses * 1e3, color=TEAL,
             lw=2.0, marker="s", markersize=4, linestyle="--", label="[0/theta/-theta/90]s (opt.)")
    ax2.set_xlabel("eta = y/b  [-]", fontsize=9)
    ax2.set_ylabel("Total thickness  h  [mm]", fontsize=9)
    ax2.set_title("Spanwise laminate thickness", fontsize=10, pad=7)
    ax2.legend(fontsize=8, framealpha=0.15, labelcolor=WHITE)
    _style(ax2)

    # ── (3) Tsai-Wu RF across span ───────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(etas, wing_A.min_rfs, color=BLUE, lw=2.0, marker="o", markersize=4,
             label="[0/+/-45/90]s")
    ax3.plot(etas, wing_B.min_rfs, color=TEAL, lw=2.0, marker="s", markersize=4,
             linestyle="--", label="[0/theta/-theta/90]s (opt.)")
    ax3.axhline(1.5, color=GOLD, lw=1.2, linestyle=":", label="RF = 1.5 target")
    ax3.axhline(1.0, color=RED,  lw=0.8, linestyle="--", alpha=0.6, label="RF = 1.0 (failure)")
    ax3.set_xlabel("eta = y/b  [-]", fontsize=9)
    ax3.set_ylabel("Min Tsai-Wu RF  [-]", fontsize=9)
    ax3.set_title("Reserve factor across span", fontsize=10, pad=7)
    ax3.legend(fontsize=8, framealpha=0.15, labelcolor=WHITE)
    _style(ax3)

    # ── (4) Ply thickness heatmap across span (Family A) ─────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    angle_labels = [f"{int(a)}deg" for a in FAM_A]
    t_matrix = wing_A.t_half_matrix * 1e3   # (n_stations, n_half) in mm

    im = ax4.imshow(
        t_matrix.T,
        aspect="auto",
        extent=[etas[0], etas[-1], -0.5, len(FAM_A) - 0.5],
        origin="lower",
        cmap="Blues",
        interpolation="bilinear",
    )
    cbar = fig.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors=WHITE, labelsize=7)
    cbar.set_label("Half-ply t [mm]", color=WHITE, fontsize=8)
    ax4.set_yticks(range(len(FAM_A)))
    ax4.set_yticklabels(angle_labels, fontsize=8)
    ax4.set_xlabel("eta = y/b  [-]", fontsize=9)
    ax4.set_title("[0/+/-45/90]s — ply thickness map", fontsize=10, pad=7)
    _style(ax4)
    cbar.outline.set_edgecolor(DIM)

    # ── Continuous theta trajectory across span (Family B overlay) ───────────
    theta_span = _np.array([r.angles_half[1] for r in wing_B.opt_results])
    ax4_twin = ax4.twinx()
    ax4_twin.plot(etas, theta_span, color=TEAL, lw=1.5, linestyle="--",
                  marker="^", markersize=4, label="opt. theta (B)")
    ax4_twin.set_ylabel("Optimal theta [deg]", fontsize=8, color=TEAL)
    ax4_twin.tick_params(colors=TEAL, labelsize=7)
    ax4_twin.set_ylim(0, 95)
    ax4_twin.spines["right"].set_edgecolor(TEAL)

    # ── Header ────────────────────────────────────────────────────────────────
    fig.text(0.5, 0.955,
             f"Wing Skin Sizing  |  Mach {MACH} @ {ALT_M/1e3:.0f} km  "
             f"|  AoA = {ALPHA_DEG} deg  |  {mat.name}  |  RF >= 1.5",
             ha="center", color=WHITE, fontsize=12, fontweight="bold")
    fig.text(0.5, 0.928,
             f"b/2={wing.semi_span}m  c_root={wing.root_chord}m  "
             f"lambda={wing.taper_ratio}  sweep={wing.sweep_le_deg}deg  "
             f"MTOW={wing.mtow_n/1e3:.0f}kN  t/c={wing.t_over_c}",
             ha="center", color=DIM, fontsize=8)

    outpath = "outputs/wing_skin_mdo.png"
    plt.savefig(outpath, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"\n  Figure saved -> {outpath}")
    plt.close()
