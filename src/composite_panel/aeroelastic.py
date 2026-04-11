"""
Iterative 1-D static aeroelastic solution for a tapered swept wing.

Euler-Bernoulli cantilever with optional D16 bend-twist coupling for
aeroelastic tailoring of unbalanced laminates.

Ref: Bisplinghoff, Ashley & Halfman (1955), Weisshaar (1981)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import aerosandbox.numpy as np

from composite_panel.aero_loads import WingGeometry, wing_panel_loads, PanelLoads, _isa as isa_atmosphere




@dataclass
class AeroelasticResult:
    """Output of static_aeroelastic().  aeroelastic_relief > 0 = tip unloaded."""
    etas:              np.ndarray
    alpha_rigid:       float
    alpha_eff:         np.ndarray
    delta_alpha:       np.ndarray
    tip_deflection:    float
    bending_slope:     np.ndarray
    EI:                np.ndarray
    loads_rigid:       List[PanelLoads]
    loads_elastic:     List[PanelLoads]
    n_iterations:      int
    converged:         bool
    aeroelastic_relief: float

    def summary(self) -> str:
        sign = "? tip loads REDUCED (washout)" if self.aeroelastic_relief > 0 else "? tip loads INCREASED (divergence)"
        return (
            f"Aeroelastic result:\n"
            f"  Rigid AoA        : {self.alpha_rigid:.2f}deg\n"
            f"  Elastic AoA tip  : {self.alpha_eff[-1]:.2f}deg  "
            f"(Deltaalpha = {self.delta_alpha[-1]:+.2f}deg)\n"
            f"  Tip deflection   : {self.tip_deflection*100:.1f} cm\n"
            f"  Aeroelastic relief: {self.aeroelastic_relief*100:+.1f}%  {sign}\n"
            f"  Iterations       : {self.n_iterations}  "
            f"({'converged' if self.converged else 'NOT CONVERGED'})"
        )




def wing_bending_stiffness(
    wing:            WingGeometry,
    laminate_A11:    float,
    etas:            np.ndarray,
    box_chord_fraction: float = 0.50,
) -> np.ndarray:
    """Spanwise EI(y) [N*m^2] from A11, thin-walled rectangular box model.
    Megson (2014), 'Aircraft Structural Analysis', Ch. 20, idealised two-boom box."""
    c   = wing.chord(etas)
    b_b = box_chord_fraction * c
    h_b = wing.t_over_c * c
    return 2.0 * laminate_A11 * b_b * (h_b / 2.0) ** 2


def wing_torsional_stiffness(
    wing:               WingGeometry,
    laminate_D66:       float,
    etas:               np.ndarray,
    box_chord_fraction: float = 0.50,
) -> np.ndarray:
    """Spanwise GJ(y) [N*m^2] from D66, closed-box Bredt model.  GJ = 4*D66*b_box.
    Megson (2014), Ch. 18, Bredt-Batho torsion for thin-walled closed sections."""
    c     = wing.chord(etas)
    b_box = box_chord_fraction * c
    return 4.0 * laminate_D66 * b_box


def wing_coupling_stiffness(
    wing:               WingGeometry,
    laminate_D16:       float,
    etas:               np.ndarray,
    box_chord_fraction: float = 0.50,
) -> np.ndarray:
    """Spanwise bend-twist coupling EK(y) [N*m^2] from D16.  EK = 2*D16*b_box.
    Weisshaar (1981), J. Aircraft, Eq. 5."""
    c     = wing.chord(etas)
    b_box = box_chord_fraction * c
    return 2.0 * laminate_D16 * b_box




def euler_bernoulli_cantilever(
    y:       np.ndarray,
    EI:      np.ndarray,
    q_dist:  np.ndarray,
) -> tuple:
    """Cantilever bending: returns (theta, w, M_beam) via trapezoid integration.
    Timoshenko & Gere (1961), 'Theory of Elastic Stability', Ch. 1."""
    n = len(y)

    # M(y_i) = integral_{y_i}^{b} q(eta)*(eta - y_i) d eta
    # Decompose: M(y_i) = Q1(y_i) - y_i * Q0(y_i)
    #   Q0(y_i) = integral_{y_i}^{b} q(eta) d eta          (shear resultant)
    #   Q1(y_i) = integral_{y_i}^{b} q(eta)*eta d eta      (first moment of load)
    # Both computed via reverse cumulative trapezoid.
    dy = np.diff(y)
    q_times_y = q_dist * y                          # q(eta_i) * eta_i  at each grid point

    Q0_inc = 0.5 * (q_dist[:-1] + q_dist[1:]) * dy
    Q1_inc = 0.5 * (q_times_y[:-1] + q_times_y[1:]) * dy

    Q0 = np.zeros(n)
    Q1 = np.zeros(n)
    Q0[:-1] = np.cumsum(Q0_inc[::-1])[::-1]
    Q1[:-1] = np.cumsum(Q1_inc[::-1])[::-1]

    M_beam = Q1 - y * Q0

    # kappa = M / EI, then theta = cumulative integral of kappa, w = cumulative integral of theta
    kappa = M_beam / np.maximum(EI, 1e-6)

    kappa_increments = 0.5 * (kappa[:-1] + kappa[1:]) * dy
    theta = np.zeros(n)
    theta[1:] = np.cumsum(kappa_increments)

    theta_increments = 0.5 * (theta[:-1] + theta[1:]) * dy
    w = np.zeros(n)
    w[1:] = np.cumsum(theta_increments)

    return theta, w, M_beam




def _build_moment_matrix(y: np.ndarray) -> np.ndarray:
    """Bending-moment influence matrix W such that M = W · q_dist.

    W[j,i] = d(M_j)/d(q_i) from trapezoid quadrature of
    M(y_j) = integral_{y_j}^{b} q(eta)*(eta - y_j) deta.
    Timoshenko & Gere (1961), Ch. 1."""
    n  = len(y)
    dy = np.diff(y)
    W  = np.zeros((n, n))
    for j in range(n):
        for k in range(j, n - 1):
            W[j, k]     += 0.5 * (y[k]     - y[j]) * dy[k]
            W[j, k + 1] += 0.5 * (y[k + 1] - y[j]) * dy[k]
    return W


def _build_cumtrap_matrix(y: np.ndarray) -> np.ndarray:
    """Forward cumulative-trapezoid matrix T such that theta = T · kappa.

    theta[0] = 0 (cantilever root BC),
    theta[j] = sum_{k=0}^{j-1} 0.5*(kappa[k]+kappa[k+1])*dy[k]."""
    n  = len(y)
    dy = np.diff(y)
    T  = np.zeros((n, n))
    for j in range(1, n):
        for k in range(j):
            T[j, k]     += 0.5 * dy[k]
            T[j, k + 1] += 0.5 * dy[k]
    return T


def static_aeroelastic(
    wing:             WingGeometry,
    mach:             float,
    altitude_m:       float,
    alpha_rigid_deg:  float,
    n_load:           float,
    laminate_A11:     float,
    laminate_h:       float,
    laminate_D16:     float = 0.0,
    laminate_D66:     Optional[float] = None,
    n_stations:       int   = 20,
    box_fraction:     float = 0.50,
) -> AeroelasticResult:
    """One-shot linear aeroelastic solve via influence-matrix assembly.

    For linear aero (Ackeret / Karman-Tsien) and linear structure
    (Euler-Bernoulli), the aeroelastic system is:
        alpha_eff = alpha_rigid + A · alpha_eff
    solved as  (I - A) · alpha_eff = alpha_rigid · 1.
    Bisplinghoff, Ashley & Halfman (1955), Ch. 8; Weisshaar (1981), J. Aircraft.
    """
    n    = n_stations
    etas = np.linspace(0.02, 0.98, n)
    y    = etas * wing.semi_span        # physical span coordinate [m]
    c    = wing.chord(etas)             # local chord [m]

    sweep_rad = np.radians(wing.sweep_le_deg)
    tan_sweep = np.tan(sweep_rad)

    # == Atmosphere ============================================================
    rho, a_sound = isa_atmosphere(altitude_m)

    EI = wing_bending_stiffness(wing, laminate_A11, etas, box_fraction)

    # == Bend-twist coupling from unbalanced laminate (D16 != 0) ===============
    bt_ratio = np.zeros(n)
    if abs(laminate_D16) > 0.0:
        if laminate_D66 is None:
            warnings.warn(
                "laminate_D16 is non-zero but laminate_D66 was not supplied. "
                "Bend-twist coupling contribution will be ignored.  "
                "Pass laminate_D66=Laminate.D[2,2] to activate aeroelastic tailoring.",
                stacklevel=2,
            )
        else:
            GJ = wing_torsional_stiffness(wing, laminate_D66, etas, box_fraction)
            EK = wing_coupling_stiffness(wing, laminate_D16, etas, box_fraction)
            bt_ratio = np.where(np.abs(GJ) > 1e-30, EK / GJ, 0.0)

    # == Aerodynamic load sensitivity  S_deg[i] = d(q_lift_i)/d(alpha_deg) ====
    # Both Ackeret and KT are linear in alpha, so evaluate at 1 deg.
    loads_unit = [
        wing_panel_loads(wing, etas[i], mach, altitude_m, 1.0, n_load)
        for i in range(n)
    ]
    S_deg = np.array([
        -2.0 * loads_unit[i].Nyy / max(c[i], 1e-3)
        for i in range(n)
    ])   # [N/m per degree]

    # == Structural influence matrices =========================================
    W = _build_moment_matrix(y)      # M = W · q_dist
    T = _build_cumtrap_matrix(y)     # theta = T · kappa

    # Washout per unit bending slope [rad/rad]:
    #   geometric sweep washout:  Bisplinghoff et al. (1955), Sec. 8-5
    #   bend-twist coupling:      Weisshaar (1981), Eq. 8
    washout = -(tan_sweep + bt_ratio)

    # == Assemble influence matrix A ===========================================
    # Chain:  alpha_deg → S_deg → q_lift → W → M → 1/EI → kappa → T → theta
    #         → washout (rad) → degrees
    # A = (180/pi) · diag(washout) · T · diag(1/EI) · W · diag(S_deg)
    inv_EI = 1.0 / np.maximum(EI, 1e-6)
    A = (180.0 / np.pi) * (
        np.diag(washout) @ T @ np.diag(inv_EI) @ W @ np.diag(S_deg)
    )

    # == Solve (I - A) · alpha_eff = alpha_rigid · 1 ==========================
    alpha_eff = np.linalg.solve(
        np.eye(n) - A,
        np.full(n, alpha_rigid_deg),
    )
    delta_alpha_deg = alpha_eff - alpha_rigid_deg

    # == Recover deflection from converged alpha ===============================
    q_lift = S_deg * alpha_eff                                   # [N/m]
    theta, w_deflect, _ = euler_bernoulli_cantilever(y, EI, q_lift)

    # == Compute panel loads at rigid and elastic alpha ========================
    loads_rigid = [
        wing_panel_loads(wing, etas[i], mach, altitude_m, alpha_rigid_deg, n_load)
        for i in range(n)
    ]
    loads_elastic = [
        wing_panel_loads(wing, etas[i], mach, altitude_m,
                         float(alpha_eff[i]), n_load)
        for i in range(n)
    ]

    # Aeroelastic relief metric (Nyy-based, free-end tip)
    Nyy_rigid_tip   = loads_rigid[-1].Nyy
    Nyy_elastic_tip = loads_elastic[-1].Nyy
    if abs(Nyy_rigid_tip) > 1.0:
        relief = (Nyy_elastic_tip - Nyy_rigid_tip) / abs(Nyy_rigid_tip)
    else:
        relief = 0.0

    return AeroelasticResult(
        etas              = etas,
        alpha_rigid       = alpha_rigid_deg,
        alpha_eff         = alpha_eff,
        delta_alpha       = delta_alpha_deg,
        tip_deflection    = float(w_deflect[-1]),
        bending_slope     = theta,
        EI                = EI,
        loads_rigid       = loads_rigid,
        loads_elastic     = loads_elastic,
        n_iterations      = 1,
        converged         = True,
        aeroelastic_relief= relief,
    )


if __name__ == "__main__":
    import sys as _sys
    _sys.stdout.reconfigure(encoding="utf-8")
    wing = WingGeometry(
        semi_span    = 4.5,     # m   --  half-span
        root_chord   = 2.0,     # m
        taper_ratio  = 0.3,     # tip/root chord
        sweep_le_deg = 45.0,    # deg  leading-edge sweep
        t_over_c     = 0.04,    # structural box depth fraction
        mtow_n       = 150_000.0,  # N  max take-off weight
    )

    # IM7/8552 [0/45/-45/90]s 8-ply laminate (h ~ 1 mm)
    # Ex ~ 52 GPa  ->  A11 = Ex * h = 52e9 * 1e-3 = 52e6 N/m
    laminate_A11 = 52e6   # N/m
    laminate_h   = 1.0e-3  # m

    result = static_aeroelastic(
        wing             = wing,
        mach             = 1.7,
        altitude_m       = 15_000.0,
        alpha_rigid_deg  = 3.5,
        n_load           = 2.5,
        laminate_A11     = laminate_A11,
        laminate_h       = laminate_h,
        n_stations       = 20,
        box_fraction     = 0.50,
    )
    print(result.summary())
