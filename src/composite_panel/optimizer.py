"""
Minimum-mass symmetric laminate sizing via IPOPT/CasADi (AeroSandbox Opti).
Tsai-Wu + optional buckling constraints.  Single-case, multi-case, wing-level,
and aeroelastic tailoring variants.
"""

import aerosandbox as asb
import aerosandbox.numpy as np      # CasADi-compatible drop-in
import numpy as _np                 # standard numpy for constants
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

from composite_panel.ply import PlyMaterial, Ply, IM7_8552
from composite_panel.laminate import Laminate
from composite_panel.failure import check_laminate
from composite_panel.aero_loads import (
    WingGeometry, wing_panel_loads, PanelLoads,
)
from composite_panel.loads_db import LoadCase, LoadsDatabase
from composite_panel.thermal import (
    PlyThermal, ThermalState, IM7_8552_thermal,
    alpha_bar as _alpha_bar, thermal_resultants as _thermal_resultants,
)
from composite_panel.buckling import (
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
# Analytical Q_bar  --  symbolic-compatible (no T^-1)
# ---------------------------------------------------------------------------

def _Q_bar_matrix(mat: PlyMaterial, angle_rad):
    """Rotated stiffness Q_bar via invariant polynomial form.  CasADi-compatible."""
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
    """Stress transformation to ply axes.  CasADi-compatible."""
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
    """A, D from symmetric half-stack.  CasADi-compatible."""
    n      = len(t_half)
    h_half = sum(t_half)           # CasADi scalar if t_half contains opti vars

    # A  --  linear in t
    A = Q_bars_half[0] * (2.0 * t_half[0])
    for k in range(1, n):
        A = A + Q_bars_half[k] * (2.0 * t_half[k])

    # D  --  cubic in t via z-coordinates; accumulate z from bottom face
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
    """Mid-plane z for all 2n plies of a symmetric stack."""
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
    """Branch-free Tsai-Wu RF.  CasADi-compatible."""
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
    """Auto-detect +/-theta index pairs in a half-stack.  Skips 0 and 90."""
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
    allow_unbalanced: bool  = False,
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
    """Minimum-mass symmetric laminate.  Tsai-Wu + optional buckling constraints."""
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

        # Balance: theta_j = -theta_i  (skipped for unbalanced aeroelastic tailoring)
        if balance_pairs is not None and not allow_unbalanced:
            for (i, j) in balance_pairs:
                opti.subject_to(theta_list[j] == -theta_list[i])
    else:
        theta_list = [_np.radians(a) for a in angles_half_deg]

    # Balance: t_i = t_j  (skipped for unbalanced aeroelastic tailoring)
    if balance_pairs is not None and not allow_unbalanced:
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
        opti.subject_to(rf_k >= rf_min + 1e-3)

    # -- Buckling constraint (differentiable through D_mat) ------------------
    if panel_a is not None and panel_b is not None:
        # Use suggest_mode_number on the initial laminate to account for D22/D11
        # anisotropy, not just panel aspect ratio.  m_x is fixed once at build
        # time -- D11/D22 may change during the solve but the error is small.
        _ang0 = angles_half_deg + list(reversed(angles_half_deg))
        _D0   = Laminate([Ply(mat, t_init, a) for a in _ang0]).D
        m_x   = suggest_mode_number(panel_a, panel_b, _D0)
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
    """Multi-case result with governing_case per ply and rf_per_case."""
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
            lines.append(f"    ply {k:2d} -> {name}")
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
    """Minimum-mass laminate sized for all load cases simultaneously."""
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

    # == Design variables: ply thicknesses ====================================
    t      = opti.variable(n_vars=n_half, init_guess=t_init, lower_bound=t_min)
    t_list = [t[k] for k in range(n_half)]

    # == Fixed angles (angle optimization not implemented for multi-case) ======
    theta_list  = [_np.radians(a) for a in angles_half_deg]
    Q_bars_half = [_Q_bar_matrix(mat, theta_list[k]) for k in range(n_half)]
    theta_full  = theta_list + list(reversed(theta_list))
    Q_bars_full = Q_bars_half + list(reversed(Q_bars_half))

    # == Balance: t_i == t_j ==================================================?
    if balance_pairs is not None:
        for (i, j) in balance_pairs:
            opti.subject_to(t_list[i] == t_list[j])

    # == ABD assembly  --  shared across all load cases ==========================?
    A_mat, D_mat = _build_ABD_symmetric(t_list, Q_bars_half)
    a_comp = np.linalg.inv(A_mat)
    d_comp = np.linalg.inv(D_mat)
    z_mids = _ply_zmid_symmetric(t_list)

    # Pre-compute buckling mode number from initial laminate (accounts for D22/D11)
    if panel_a is not None and panel_b is not None:
        _ang0_mc = angles_half_deg + list(reversed(angles_half_deg))
        _D0_mc   = Laminate([Ply(mat, t_init, a) for a in _ang0_mc]).D
        _m_x_mc  = suggest_mode_number(panel_a, panel_b, _D0_mc)

    # == Tsai-Wu constraints for EVERY load case ==============================?
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
            opti.subject_to(rf_k >= rf_min + 1e-3)

        # == Optional buckling constraint per case ============================?
        if panel_a is not None and panel_b is not None:
            RF_b = _buckling_rf_smooth(
                N_c[0], N_c[1], N_c[2],
                D_mat, panel_a, panel_b, _m_x_mc, 1,
            )
            opti.subject_to(RF_b >= buckle_rf_min)

    # == Objective: minimum areal mass ========================================?
    opti.minimize(rho_kg_m3 * 2.0 * sum(t_list))

    # == Solve ================================================================?
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

    # == Post-process: standard numpy to find governing case per ply ==========
    t_full_opt = _np.concatenate([t_half_opt, t_half_opt[::-1]])
    ang_full   = ang_half_opt + list(reversed(ang_half_opt))
    h_opt      = float(t_full_opt.sum())

    lam_chk = Laminate([Ply(mat, t_full_opt[k], ang_full[k]) for k in range(n_full)])

    # Track min RF per ply per case  --  find which case governs each ply
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
    Station-by-station laminate sizing across the wing semi-span.
    Runs optimize_laminate() independently at each eta station.
    """
    etas = _np.linspace(0.05, 0.95, n_stations)

    # == Pass 1: aerodynamic loads at every station ==========================
    loads_list = [
        wing_panel_loads(wing, eta, mach, altitude_m, alpha_deg, n_load)
        for eta in etas
    ]

    # == Torsion pre-pass (Bredt-Batho shear flow from AC/EA offset) ========?
    # Pitching moment per unit span: m'(y) = delta_p(y) * c(y)^2 * ea_offset
    # delta_p recoverd from Nyy = -delta_p * c/2  ->  delta_p = -2*Nyy/c
    # Torque: T(y) = integral_y^{b/2} m'(y') dy'  (cumulative sum from tip inward)
    # Enclosed box area: A_box ~= 0.5 * t/c * c^2  (height=t/c*c, width~=c/2)
    # Shear flow: Nxy_torsion = T / (2 * A_box)
    if wing.ea_offset != 0.0:
        y_arr   = etas * wing.semi_span
        m_prime = _np.array([
            (-2.0 * loads_list[i].Nyy / wing.chord(etas[i]))
            * wing.chord(etas[i]) ** 2 * wing.ea_offset
            for i in range(n_stations)
        ])
        T = _np.zeros(n_stations)
        for i in range(n_stations - 2, -1, -1):
            T[i] = T[i + 1] + 0.5 * (m_prime[i] + m_prime[i + 1]) * (y_arr[i + 1] - y_arr[i])
        for i in range(n_stations):
            c_i   = wing.chord(etas[i])
            A_box = 0.5 * wing.t_over_c * c_i ** 2
            loads_list[i].Nxy += T[i] / (2.0 * A_box)

    # == Pass 2: optimize laminate at each station ============================
    opt_list = []

    print(f"  Sizing {n_stations} spanwise stations  --  RF >= {rf_min}  t_min = {t_min*1e3:.2f} mm")

    for i, eta in enumerate(etas):
        panel_loads = loads_list[i]
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
# Aeroelastic tailoring optimiser
# ---------------------------------------------------------------------------

@dataclass
class AeroelasticOptimizationResult:
    """Laminate optimisation result with embedded aeroelastic washout constraint."""
    base:                 OptimizationResult
    achieved_washout_deg: float
    relief_min_target:    float
    D16:                  float
    D66:                  float
    EI_root:              float
    GJ_root:              float
    EK_root:              float
    bt_ratio_root:        float

    @property
    def total_h(self):        return self.base.total_h
    @property
    def areal_density(self):  return self.base.areal_density
    @property
    def min_tsai_wu_rf(self): return self.base.min_tsai_wu_rf
    @property
    def converged(self):      return self.base.converged
    @property
    def t_half(self):         return self.base.t_half
    @property
    def t_full(self):         return self.base.t_full
    @property
    def angles_half(self):    return self.base.angles_half

    def summary(self) -> str:
        lines = [
            self.base.summary(),
            "  Aeroelastic tailoring",
            f"    washout target   : <= {-abs(self.relief_min_target):.3f}deg  (nose-down)",
            f"    washout achieved : {self.achieved_washout_deg:+.4f}deg",
            f"    D16              : {self.D16:.4f} N*m",
            f"    D66              : {self.D66:.4f} N*m",
            f"    EK/GJ (root)     : {self.bt_ratio_root:.5f}",
            f"    EI root          : {self.EI_root:.3e} N*m^2",
        ]
        return "\n".join(lines)


def optimize_laminate_aeroelastic(
    N_loads:          _np.ndarray,
    M_loads:          _np.ndarray,
    mat:              PlyMaterial,
    angles_half_deg:  List[float],
    wing:             "WingGeometry",
    n_load:           float,
    relief_min_deg:   float,
    use_bt_coupling:  bool  = True,
    box_fraction:     float = 0.50,
    rho_kg_m3:        float = 1600.0,
    rf_min:           float = 1.5,
    t_min:            float = 0.05e-3,
    t_init:           float = 0.125e-3,
    angle_bounds_deg: Optional[List[Tuple[float,float]]] = None,
    panel_a:          Optional[float] = None,
    panel_b:          Optional[float] = None,
    buckle_rf_min:    float = 1.0,
    thermal_state:    Optional[ThermalState]  = None,
    ply_thermal:      Optional[PlyThermal]    = None,
    verbose:          bool  = True,
) -> AeroelasticOptimizationResult:
    """
    Minimum-mass laminate with CasADi-native aeroelastic washout constraint.
    EI/GJ/EK derived from A/D matrices -- fully differentiable through IPOPT.
    """
    import math as _math

    n_half = len(angles_half_deg)
    N_vec  = np.array(N_loads, dtype=float)
    M_vec  = np.array(M_loads, dtype=float)

    if thermal_state is not None:
        pt = ply_thermal if ply_thermal is not None else IM7_8552_thermal()
        _ang_full_init = angles_half_deg + list(reversed(angles_half_deg))
        _lam_init = Laminate([Ply(mat, t_init, a) for a in _ang_full_init])
        _N_T, _M_T = _thermal_resultants(
            _lam_init.plies, [pt] * (2 * n_half),
            thermal_state, _lam_init.z_interfaces,
        )
        N_vec = N_vec + _N_T
        M_vec = M_vec + _M_T

    if angle_bounds_deg is None:
        angle_bounds_deg = [(-90.0, 90.0)] * n_half

    opti = asb.Opti()

    # == Thickness variables ====================================================
    t      = opti.variable(n_vars=n_half, init_guess=t_init, lower_bound=t_min)
    t_list = [t[k] for k in range(n_half)]

    # == Angle variables (if bend-twist coupling is requested) ================?
    if use_bt_coupling:
        theta_init = _np.radians([a for a in angles_half_deg])
        lo = _np.radians([b[0] for b in angle_bounds_deg])
        hi = _np.radians([b[1] for b in angle_bounds_deg])
        theta = opti.variable(n_vars=n_half, init_guess=theta_init,
                              lower_bound=lo, upper_bound=hi)
        theta_list = [theta[k] for k in range(n_half)]
        # No balance constraints  --  allow D16 ? 0
    else:
        theta_list = [_np.radians(a) for a in angles_half_deg]

    # == ABD assembly ==========================================================?
    Q_bars_half = [_Q_bar_matrix(mat, theta_list[k]) for k in range(n_half)]
    Q_bars_full = Q_bars_half + list(reversed(Q_bars_half))
    theta_full  = theta_list + list(reversed(theta_list))

    A_mat, D_mat = _build_ABD_symmetric(t_list, Q_bars_half)
    a_comp = np.linalg.inv(A_mat)
    d_comp = np.linalg.inv(D_mat)
    z_mids = _ply_zmid_symmetric(t_list)

    # == Tsai-Wu constraints ====================================================
    eps0  = a_comp @ N_vec
    kappa = d_comp @ M_vec

    for k in range(2 * n_half):
        Qb    = Q_bars_full[k]
        T_k   = _T_stress(theta_full[k])
        z_k   = z_mids[k]
        eps_xy = eps0 + z_k * kappa
        sig_xy = Qb  @ eps_xy
        sig_12 = T_k @ sig_xy
        rf_k   = _tsai_wu_rf(sig_12[0], sig_12[1], sig_12[2], mat)
        opti.subject_to(rf_k >= rf_min + 1e-3)

    # == Buckling constraint ====================================================
    if panel_a is not None and panel_b is not None:
        # Use suggest_mode_number on initial laminate (see optimize_laminate)
        _ang0_ae = angles_half_deg + list(reversed(angles_half_deg))
        _D0_ae   = Laminate([Ply(mat, t_init, a) for a in _ang0_ae]).D
        m_x      = suggest_mode_number(panel_a, panel_b, _D0_ae)
        RF_buckle = _buckling_rf_smooth(
            N_vec[0], N_vec[1], N_vec[2],
            D_mat, panel_a, panel_b, m_x, 1,
        )
        opti.subject_to(RF_buckle >= buckle_rf_min)

    # == Aeroelastic washout constraint (CasADi-native) ========================?
    #
    # Wing-box dimensions at root chord (constant, not design variables):
    #   b_box = box_fraction * c_root  [m]
    #   h_box = t_over_c    * c_root  [m]
    #
    # Bending, torsional, and coupling stiffnesses from CLT:
    #   EI    = 2 * A11 * b_box * (h_box/2)^2   [N*m^2]  (A11=E_eff*h cancels h)
    #   GJ    = 4 * D66 * b_box                  [N*m^2]
    #   EK    = 2 * D16 * b_box                  [N*m^2]  (zero for balanced)
    #
    # Tip slope (elliptic spanwise lift, derived via double integration):
    #   theta_tip = W_semi * L^2 / (8 * EI)     [rad]
    #   Derivation: for q(y)=q0*sqrt(1-(y/L)^2), integral q*y^2/2 dy over [0,L]
    #   gives theta_tip = (pi*q0*L^3/32)/EI = W*L^2/(8*EI)  (exact for elliptic).
    #
    # Total tip washout (sweep geometry + bend-twist coupling):
    #   Deltaalpha_tip = ?theta_tip * (tan Lambda + EK/GJ)  [rad]
    #
    # Constraint: Deltaalpha_tip [deg] <= ?|relief_min_deg|
    #
    c_root    = wing.root_chord
    b_box     = box_fraction * c_root
    h_box     = wing.t_over_c * c_root
    L         = wing.semi_span
    W_semi    = wing.mtow_n * n_load * 0.5
    tan_sweep = _math.tan(_math.radians(wing.sweep_le_deg))

    A11       = A_mat[0, 0]                                  # CasADi expression
    D16       = D_mat[0, 2]                                  # zero for balanced
    D66       = D_mat[2, 2]

    EI        = 2.0 * A11 * b_box * (h_box * 0.5) ** 2      # [N*m^2]
    GJ        = 4.0 * D66 * b_box                            # [N*m^2]
    EK        = 2.0 * D16 * b_box                            # [N*m^2]

    theta_tip = W_semi * L ** 2 / (8.0 * (EI + 1e-3))       # [rad]  eps for stability
    bt_ratio  = EK / (GJ + 1e-6)
    delta_alpha_rad = -theta_tip * (tan_sweep + bt_ratio)
    delta_alpha_deg_expr = delta_alpha_rad * (180.0 / _math.pi)

    opti.subject_to(delta_alpha_deg_expr <= -abs(relief_min_deg))

    # == Objective ============================================================?
    opti.minimize(rho_kg_m3 * 2.0 * sum(t_list))

    # == Solve ================================================================?
    try:
        sol       = opti.solve(verbose=verbose)
        converged = True
        t_half_opt = _np.array([float(sol(t[k])) for k in range(n_half)])
        if use_bt_coupling:
            ang_half_opt = [float(_np.degrees(sol(theta[k]))) for k in range(n_half)]
        else:
            ang_half_opt = list(angles_half_deg)

        # Extract aeroelastic quantities at optimum
        D16_opt = float(sol(D16))
        D66_opt = float(sol(D66))
        EI_opt  = float(sol(EI))
        GJ_opt  = float(sol(GJ))
        EK_opt  = float(sol(EK))
        bt_opt  = float(sol(bt_ratio))
        washout_opt = float(sol(delta_alpha_deg_expr))

    except Exception as exc:
        if verbose:
            print(f"  Optimizer did not converge: {exc}")
        converged    = False
        t_half_opt   = _np.full(n_half, t_init)
        ang_half_opt = list(angles_half_deg)
        D16_opt = D66_opt = EI_opt = GJ_opt = EK_opt = bt_opt = 0.0
        washout_opt = 0.0

    # == Post-process ==========================================================
    t_full_opt  = _np.concatenate([t_half_opt, t_half_opt[::-1]])
    ang_full    = ang_half_opt + list(reversed(ang_half_opt))
    h_opt       = float(t_full_opt.sum())

    lam_chk = Laminate([Ply(mat, t_full_opt[k], ang_full[k]) for k in range(2 * n_half)])
    resp    = lam_chk.response(N=N_vec, M=M_vec)
    fails   = check_laminate(resp, lam_chk.plies, criterion="tsai_wu", verbose=False)
    min_rf  = float(min(r.rf for r in fails))

    # -- Buckling verification: exact post-check against target ----------------
    # The smooth formula uses fixed mode numbers and an algebraic D16 knockdown.
    # After convergence, re-evaluate with the exact formula (full mode sweep,
    # D16 coupling warning) to catch cases where the optimizer constraint was
    # met but the exact critical load is below the target.
    if panel_a is not None and panel_b is not None and converged:
        rf_exact = _buckling_rf(_np.array(N_loads), lam_chk.D, panel_a, panel_b)
        if rf_exact < buckle_rf_min:
            import warnings as _warnings
            _warnings.warn(
                f"Post-optimisation buckling check FAILED: exact RF={rf_exact:.3f} "
                f"< target {buckle_rf_min:.3f}.  Smooth optimizer constraint was "
                f"satisfied but exact mode-sweep disagrees.  "
                f"D16={lam_chk.D[0,2]:.1f} N*m -- consider FEA validation.",
                stacklevel=2,
            )

    base = OptimizationResult(
        angles_half    = ang_half_opt,
        t_half         = t_half_opt,
        t_full         = t_full_opt,
        total_h        = h_opt,
        areal_density  = rho_kg_m3 * h_opt,
        min_tsai_wu_rf = min_rf,
        rf_min_target  = rf_min,
        converged      = converged,
    )

    return AeroelasticOptimizationResult(
        base                 = base,
        achieved_washout_deg = washout_opt,
        relief_min_target    = abs(relief_min_deg),
        D16                  = D16_opt,
        D66                  = D66_opt,
        EI_root              = EI_opt,
        GJ_root              = GJ_opt,
        EK_root              = EK_opt,
        bt_ratio_root        = bt_opt,
    )


# ---------------------------------------------------------------------------
# Demo  --  python optimize_laminate.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import matplotlib.cm as cm

    # == Palette ============================================================?
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

    # == Wing definition  (representative supersonic fighter class) ==========?
    wing = WingGeometry(
        semi_span    = 4.5,       # m   --  ~9 m full span
        root_chord   = 4.0,       # m
        taper_ratio  = 0.25,      # cropped delta planform
        sweep_le_deg = 50.0,      # deg   --  strongly swept leading edge
        t_over_c     = 0.04,      # 4% thin supersonic profile
        mtow_n       = 120_000.0, # N    --  ~12 tonne class
    )

    # == Flight condition ====================================================?
    MACH      = 1.6
    ALT_M     = 15_000
    ALPHA_DEG = 4.0
    N_LOAD    = 2.5
    RHO_PLY   = 1600.0   # kg/m3  IM7/8552 cured density
    mat       = IM7_8552()

    # == Layup families ======================================================?
    # Family A: [0/+/-45/90]s   --  standard aerospace quasi-isotropic
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

    # == Wing sizing: Family A ================================================
    print("\nFamily A  [0/+/-45/90]s  --  spanwise sizing ...")
    wing_A = optimize_wing(
        wing=wing, mach=MACH, altitude_m=ALT_M, alpha_deg=ALPHA_DEG,
        mat=mat, angles_half_deg=FAM_A,
        n_load=N_LOAD, n_stations=12, rf_min=1.5, t_min=0.05e-3, t_init=0.15e-3,
        balance_pairs=PAIRS_A, rho_kg_m3=RHO_PLY,
    )

    # == Wing sizing: Family B (continuous theta) ============================?
    print("\nFamily B  [0/theta/-theta/90]s continuous  --  spanwise sizing ...")
    wing_B = optimize_wing(
        wing=wing, mach=MACH, altitude_m=ALT_M, alpha_deg=ALPHA_DEG,
        mat=mat, angles_half_deg=FAM_B_INIT,
        n_load=N_LOAD, n_stations=12, rf_min=1.5, t_min=0.05e-3, t_init=0.15e-3,
        balance_pairs=PAIRS_B, optimize_angles=True, angle_bounds_deg=BOUNDS_B,
        rho_kg_m3=RHO_PLY,
    )

    # == Figure: 4-panel wing skin map ========================================
    fig = plt.figure(figsize=(15, 10), facecolor=BG)
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.44, wspace=0.30,
                            left=0.08, right=0.97, top=0.89, bottom=0.09)

    etas = wing_A.etas

    # == (1) Spanwise load distribution ======================================
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

    # == (2) Spanwise total thickness ========================================?
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

    # == (3) Tsai-Wu RF across span ==========================================?
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

    # == (4) Ply thickness heatmap across span (Family A) ====================?
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
    ax4.set_title("[0/+/-45/90]s  --  ply thickness map", fontsize=10, pad=7)
    _style(ax4)
    cbar.outline.set_edgecolor(DIM)

    # == Continuous theta trajectory across span (Family B overlay) ==========?
    theta_span = _np.array([r.angles_half[1] for r in wing_B.opt_results])
    ax4_twin = ax4.twinx()
    ax4_twin.plot(etas, theta_span, color=TEAL, lw=1.5, linestyle="--",
                  marker="^", markersize=4, label="opt. theta (B)")
    ax4_twin.set_ylabel("Optimal theta [deg]", fontsize=8, color=TEAL)
    ax4_twin.tick_params(colors=TEAL, labelsize=7)
    ax4_twin.set_ylim(0, 95)
    ax4_twin.spines["right"].set_edgecolor(TEAL)

    # == Header ================================================================
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
