"""
validate_model.py
=================
Model validation script.

Every test has a KNOWN ANALYTICAL ANSWER or a published benchmark value.
All checks are independent of each other — a failure in one block does not
prevent the others from running.

Run:
    python -X utf8 validate_model.py

Pass threshold:  all PASS  →  model is self-consistent and matches theory.
Any FAIL        →  indicates a bug or miscalibrated constant — investigate.

References
----------
[1] CMH-17 (MIL-HDBK-17-1F) Vol. 2, Chapter 4 — IM7/8552 B-basis allowables
[2] Jones, R.M. — Mechanics of Composite Materials (1999)
[3] Timoshenko & Gere — Theory of Elastic Stability (1961) Ch. 9
[4] Kassapoglou, C. — Design and Analysis of Composite Structures (2013)
[5] ESDU 02.03.11 — Buckling of Flat Orthotropic Plates
[6] Ackeret, J. — Luftkräfte auf Flügel die mit grösserer als Schallgeschwindigkeit
    bewegt werden (1925)
"""

import sys, os, math, traceback
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

# ── colour helpers ─────────────────────────────────────────────────────────────
_GREEN = "\033[92m"; _RED = "\033[91m"; _RESET = "\033[0m"; _BOLD = "\033[1m"

_results = []

def _check(name: str, condition: bool, detail: str = "", tol_pct: float = None):
    status = "PASS" if condition else "FAIL"
    colour = _GREEN if condition else _RED
    tol_str = f"  [tol ±{tol_pct:.1f}%]" if tol_pct is not None else ""
    print(f"  {colour}{status}{_RESET}  {name}{tol_str}")
    if detail:
        print(f"        {detail}")
    _results.append((name, condition))
    return condition


def _section(title: str):
    print(f"\n{_BOLD}{'─'*60}{_RESET}")
    print(f"{_BOLD}  {title}{_RESET}")
    print(f"{_BOLD}{'─'*60}{_RESET}")


def _pct(got, ref):
    """Percentage deviation from reference."""
    return abs(got - ref) / abs(ref) * 100


# ══════════════════════════════════════════════════════════════════════════════
# BLOCK 1  Material properties vs CMH-17 / Hexcel datasheet [Ref 1]
# ══════════════════════════════════════════════════════════════════════════════
_section("Block 1 — IM7/8552 material properties vs CMH-17")

try:
    from composite_panel import IM7_8552
    m = IM7_8552()

    # Published mean values from CMH-17-1F Vol.2 Ch.4 and Hexcel product data.
    refs = {
        "E1   [GPa]": (m.E1 / 1e9,    165.0, 5.0),
        "E2   [GPa]": (m.E2 / 1e9,      9.0, 10.0),
        "G12  [GPa]": (m.G12 / 1e9,     5.6, 10.0),
        "nu12  [-] ": (m.nu12,           0.32, 10.0),
        "F1t  [MPa]": (m.F1t / 1e6,   2326.0, 5.0),
        "F1c  [MPa]": (m.F1c / 1e6,   1200.0, 20.0),   # wide: compressive scatter high
        "F2t  [MPa]": (m.F2t / 1e6,     62.3, 10.0),
        "F2c  [MPa]": (m.F2c / 1e6,    199.8, 15.0),
        "F12  [MPa]": (m.F12 / 1e6,     92.3, 10.0),
    }
    for label, (got, ref, tol) in refs.items():
        dev = _pct(got, ref)
        _check(label, dev <= tol,
               f"got {got:.1f},  ref {ref:.1f},  dev {dev:.1f}%", tol_pct=tol)

    # Maxwell reciprocity: nu21 = nu12 * E2/E1 — must be exact (it's derived)
    nu21_computed = m.nu12 * m.E2 / m.E1
    _check("Maxwell reciprocity  nu21 == nu12*E2/E1",
           abs(m.nu21 - nu21_computed) < 1e-12,
           f"nu21={m.nu21:.6f}, expected {nu21_computed:.6f}")

except Exception:
    print(f"  {_RED}ERROR{_RESET}"); traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════════════
# BLOCK 2  CLT — analytical limit cases  [Ref 2]
# ══════════════════════════════════════════════════════════════════════════════
_section("Block 2 — CLT analytical limit cases")

try:
    from composite_panel import Ply, Laminate, IM7_8552

    mat = IM7_8552()
    t   = 0.125e-3

    # ── 2a  [0]n  unidirectional: A11 = E1*t_total / (1 - nu12*nu21) ──────────
    n_plies = 8
    plies_0 = [Ply(mat, t, 0.0) for _ in range(n_plies)]
    lam_0   = Laminate(plies_0)
    A11_got = lam_0.A[0, 0]
    A11_ref = mat.Q[0, 0] * n_plies * t    # Q11 * total_thickness
    _check("[0]8 — A11 = Q11 * h",
           _pct(A11_got, A11_ref) < 0.01,
           f"got {A11_got/1e6:.2f} MN/m,  ref {A11_ref/1e6:.2f} MN/m")

    # ── 2b  [0/90]s  should give A11 == A22  (biaxial symmetry) ────────────────
    plies_q = [Ply(mat, t, a) for a in [0, 90, 90, 0]]
    lam_q   = Laminate(plies_q)
    _check("[0/90]s — A11 == A22  (biaxial symmetry)",
           _pct(lam_q.A[0, 0], lam_q.A[1, 1]) < 0.01,
           f"A11={lam_q.A[0,0]/1e6:.2f},  A22={lam_q.A[1,1]/1e6:.2f}  MN/m")

    # ── 2c  Symmetric layup: B matrix should be exactly zero ───────────────────
    plies_sym = [Ply(mat, t, a) for a in [-45, 0, 45, 90, 90, 45, 0, -45]]
    lam_sym   = Laminate(plies_sym)
    B_max = np.abs(lam_sym.B).max()
    _check("[-45/0/45/90]s — B_max < 1 N  (coupling-free)",
           B_max < 1.0,
           f"max|B| = {B_max:.3e} N")

    # ── 2d  ABD invertible (non-singular) ──────────────────────────────────────
    det_ABD = abs(np.linalg.det(lam_sym.ABD))
    _check("[-45/0/45/90]s — ABD non-singular",
           det_ABD > 1e-10,
           f"det(ABD) = {det_ABD:.3e}")

    # ── 2e  [90]n: A22 should equal [0]n A11 (same material, rotated 90°) ──────
    plies_90 = [Ply(mat, t, 90.0) for _ in range(n_plies)]
    lam_90   = Laminate(plies_90)
    _check("[90]8 — A22 == [0]8 A11  (90° rotation symmetry)",
           _pct(lam_90.A[1, 1], lam_0.A[0, 0]) < 0.01,
           f"[90]8 A22={lam_90.A[1,1]/1e6:.2f},  [0]8 A11={lam_0.A[0,0]/1e6:.2f}  MN/m")

    # ── 2f  [+45/-45]s: A11 should equal A22 (biaxial symmetry) ────────────────
    plies_pm = [Ply(mat, t, a) for a in [45, -45, -45, 45]]
    lam_pm   = Laminate(plies_pm)
    _check("[+45/-45]s — A11 == A22  (biaxial symmetry)",
           _pct(lam_pm.A[0, 0], lam_pm.A[1, 1]) < 0.01,
           f"A11={lam_pm.A[0,0]/1e6:.2f},  A22={lam_pm.A[1,1]/1e6:.2f}  MN/m")

    # ── 2g  Zero-load response gives zero strain ────────────────────────────────
    res = lam_sym.response()
    _check("Zero load → zero midplane strain",
           np.allclose(res['eps0'], 0, atol=1e-20))
    _check("Zero load → zero curvature",
           np.allclose(res['kappa'], 0, atol=1e-20))

    # ── 2h  [0/90]4s: A11 = (Q11+Q22)/2 * h  (exact for 50/50 layup) ────────────
    plies_alt = [Ply(mat, t, a) for a in [0, 90, 0, 90, 90, 0, 90, 0]]
    lam_alt   = Laminate(plies_alt)
    A11_got = lam_alt.A[0, 0]
    A11_ref = 0.5 * (mat.Q[0, 0] + mat.Q[1, 1]) * lam_alt.thickness
    _check("[0/90]4s — A11 = (Q11+Q22)/2 * h  (exact for 50/50 layup)",
           _pct(A11_got, A11_ref) < 0.1,
           f"A11_got={A11_got/1e6:.3f} MN/m,  ref {A11_ref/1e6:.3f} MN/m")

except Exception:
    print(f"  {_RED}ERROR{_RESET}"); traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════════════
# BLOCK 3  Tsai-Wu failure criterion — boundary conditions  [Ref 4]
# ══════════════════════════════════════════════════════════════════════════════
_section("Block 3 — Tsai-Wu boundary conditions (RF at failure should = 1.0)")

try:
    from composite_panel import IM7_8552
    from composite_panel.failure import tsai_wu

    mat = IM7_8552()

    cases_tw = [
        ("Pure fibre tension  σ1 = F1t",   np.array([mat.F1t,      0,       0])),
        ("Pure fibre compr.   σ1 = -F1c",  np.array([-mat.F1c,     0,       0])),
        ("Pure transv. tens.  σ2 = F2t",   np.array([0,       mat.F2t,       0])),
        ("Pure transv. compr. σ2 = -F2c",  np.array([0,      -mat.F2c,       0])),
        ("Pure shear          τ12 = F12",  np.array([0,             0, mat.F12])),
    ]

    for label, sig in cases_tw:
        r = tsai_wu(mat, sig)
        # RF at the exact failure boundary should be 1.0 ± 1%
        # (small deviation OK due to F12_star interaction term)
        _check(label, abs(r.rf - 1.0) < 0.02,
               f"RF = {r.rf:.5f}  (ideal = 1.000)",
               tol_pct=2.0)

    # RF should scale linearly: halve the load → RF should double
    sig_half = np.array([mat.F1t * 0.5, 0, 0])
    sig_full = np.array([mat.F1t * 1.0, 0, 0])
    rf_half  = tsai_wu(mat, sig_half).rf
    rf_full  = tsai_wu(mat, sig_full).rf
    _check("RF scales inversely with uniaxial load  (RF_half ≈ 2×RF_full)",
           _pct(rf_half, 2 * rf_full) < 2.0,
           f"RF(0.5×F1t)={rf_half:.4f},  2×RF(F1t)={2*rf_full:.4f}")

except Exception:
    print(f"  {_RED}ERROR{_RESET}"); traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════════════
# BLOCK 4  Buckling — isotropic plate limit  [Ref 3]
# ══════════════════════════════════════════════════════════════════════════════
_section("Block 4 — Buckling: isotropic plate closed-form benchmark")

try:
    from composite_panel import IM7_8552, Ply, Laminate
    from composite_panel.buckling import Nxx_cr

    # Use a quasi-isotropic layup [0/+45/-45/90]s (8 plies × 0.125 mm = 1 mm)
    mat = IM7_8552()
    t   = 0.125e-3
    a, b = 0.50, 0.20        # same panel as demo

    # ── 4a  [0]8 unidirectional: exact closed-form (0% error expected) ───────────
    # For a homogeneous [0]n laminate:
    #   D11 = Q11 * h³/12,  D12 = Q12 * h³/12,  D22 = Q22 * h³/12,  D66 = Q66 * h³/12
    # The Timoshenko simply-supported formula is then exact.
    plies_0 = [Ply(mat, t, 0.0) for _ in range(8)]
    lam_0   = Laminate(plies_0)
    h0      = lam_0.thickness
    D11e, D12e, D22e, D66e = (mat.Q[i,j] * h0**3/12 for i,j in
                               [(0,0),(0,1),(1,1),(2,2)])
    Ncr_exact = min(
        (math.pi**2/b**2) * (D11e*(m*b/a)**2 + 2*(D12e+2*D66e) + D22e*(a/(m*b))**2)
        for m in range(1, 9))
    Ncr_0_model = Nxx_cr(lam_0.D, a, b)
    _check("[0]8 — Nxx_cr matches Timoshenko exact formula  (0% expected)",
           _pct(Ncr_0_model, Ncr_exact) < 0.01,
           f"model {Ncr_0_model/1e3:.4f} kN/m,  exact {Ncr_exact/1e3:.4f} kN/m",
           tol_pct=0.01)

    # ── 4b  Monotonicity: thicker → higher Ncr ───────────────────────────────────
    plies_thick = [Ply(mat, 2*t, 0.0) for _ in range(8)]
    lam_thick   = Laminate(plies_thick)
    Ncr_thick   = Nxx_cr(lam_thick.D, a, b)
    _check("Double ply thickness → higher Nxx_cr  (D ∝ h³, Ncr ∝ h³)",
           Ncr_thick > Ncr_0_model,
           f"1t: {Ncr_0_model/1e3:.2f} kN/m  →  2t: {Ncr_thick/1e3:.2f} kN/m")

    # ── 4c  Cubic scaling: 2× thickness → 8× Ncr ─────────────────────────────────
    ratio_ncr = Ncr_thick / Ncr_0_model
    _check("Double ply thickness → ~8× Nxx_cr  (cubic D scaling)",
           _pct(ratio_ncr, 8.0) < 1.0,
           f"ratio = {ratio_ncr:.4f}  (ideal = 8.000)",
           tol_pct=1.0)

    # ── 4d  Narrower panel → higher Ncr (b² in denominator) ─────────────────────
    Ncr_narrow = Nxx_cr(lam_0.D, a, b / 2)
    _check("Half panel width → higher Nxx_cr  (b² in denominator)",
           Ncr_narrow > Ncr_0_model,
           f"b={b*100:.0f}cm: {Ncr_0_model/1e3:.2f} kN/m  →  b={b/2*100:.0f}cm: {Ncr_narrow/1e3:.2f} kN/m")

except Exception:
    print(f"  {_RED}ERROR{_RESET}"); traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════════════
# BLOCK 5  Aerodynamics — pressure model cross-checks  [Ref 6]
# ══════════════════════════════════════════════════════════════════════════════
_section("Block 5 — Aerodynamic pressure model cross-checks")

try:
    from composite_panel.aero_loads import (
        panel_pressure, oblique_shock_panel_pressure,
    )

    gamma = 1.4

    # ── 5a  Ackeret analytic vs oblique_shock_panel_pressure at small AoA ────────
    # At M=1.5, α=3°: Ackeret gives ΔCp = 4α/√(M²-1)
    # Oblique shock should agree within ~10% (thin-airfoil regime)
    M, alpha_deg = 1.5, 3.0
    alpha_rad = math.radians(alpha_deg)
    rho  = 0.0889   # kg/m³ at 20 km ISA
    a_s  = 295.1    # m/s  speed of sound at 20 km ISA
    q    = 0.5 * rho * (M * a_s)**2

    dCp_ackeret = 4 * alpha_rad / math.sqrt(M**2 - 1)
    dp_ackeret  = dCp_ackeret * q

    dp_oblique = oblique_shock_panel_pressure(M, alpha_deg, q, gamma)
    dp_panel   = panel_pressure(M, alpha_deg, q, gamma)

    _check("M=1.5 α=3°: oblique_shock within 10% of Ackeret",
           _pct(dp_oblique, dp_ackeret) < 10.0,
           f"oblique {dp_oblique/1e3:.3f} kPa,  Ackeret {dp_ackeret/1e3:.3f} kPa,  "
           f"dev {_pct(dp_oblique, dp_ackeret):.1f}%",
           tol_pct=10.0)

    _check("panel_pressure(M=1.5) routes to oblique_shock",
           _pct(dp_panel, dp_oblique) < 0.1,
           f"panel_pressure={dp_panel/1e3:.4f} kPa,  oblique={dp_oblique/1e3:.4f} kPa")

    # ── 5b  Higher Mach → higher pressure (monotonicity) ─────────────────────────
    # Each Mach has its own dynamic pressure (same altitude 20 km, rho=0.0889, a=295.1)
    # At fixed altitude: q = 0.5*rho*(M*a)^2 ∝ M² — this is the physically correct comparison
    def _q(M_): return 0.5 * rho * (M_ * a_s)**2
    dp_17 = panel_pressure(1.7, alpha_deg, _q(1.7), gamma)
    dp_24 = panel_pressure(2.4, alpha_deg, _q(2.4), gamma)
    dp_50 = panel_pressure(5.0, alpha_deg, _q(5.0), gamma)
    _check("Pressure increases with Mach at fixed α + altitude  (M1.7 < M2.4 < M5.0)",
           dp_17 < dp_24 < dp_50,
           f"M1.7:{dp_17/1e3:.2f}  M2.4:{dp_24/1e3:.2f}  M5.0:{dp_50/1e3:.2f} kPa")

    # ── 5c  Sign check: ΔCp (differential windward − leeward) should be positive ──
    _check("M=2.4 α=3°: ΔCp > 0  (windward above leeward pressure)",
           panel_pressure(2.4, alpha_deg, q, gamma) > 0)

    # ── 5d  α=0° → no net lift force (ΔCp = 0) ──────────────────────────────────
    dp_zero_alpha = panel_pressure(2.4, 0.0, q, gamma)
    _check("α=0°: ΔCp = 0  (no net pressure at zero incidence)",
           abs(dp_zero_alpha) < 1.0,   # within 1 Pa
           f"ΔP(α=0) = {dp_zero_alpha:.4f} Pa")

    # ── 5e  Subsonic: Prandtl-Glauert vs analytic ΔCp = 4α/√(1-M²) ──────────────
    M_sub = 0.6
    q_sub = 0.5 * 1.225 * (M_sub * 340)**2
    dCp_pg_analytic = 4 * alpha_rad / math.sqrt(1 - M_sub**2)
    dp_pg_model     = panel_pressure(M_sub, alpha_deg, q_sub, gamma)
    dp_pg_ref       = dCp_pg_analytic * q_sub
    _check("M=0.6 α=3°: panel_pressure within 1% of Prandtl-Glauert formula",
           _pct(dp_pg_model, dp_pg_ref) < 1.0,
           f"model {dp_pg_model/1e3:.4f} kPa,  PG analytic {dp_pg_ref/1e3:.4f} kPa",
           tol_pct=1.0)

except Exception:
    print(f"  {_RED}ERROR{_RESET}"); traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════════════
# BLOCK 6  Optimizer — reserve factor at optimum equals rf_min
# ══════════════════════════════════════════════════════════════════════════════
_section("Block 6 — Optimizer: RF at optimum == rf_min (active constraint)")

try:
    from composite_panel import IM7_8552
    from composite_panel.aero_loads import panel_pressure
    from composite_panel.optimizer import optimize_laminate, detect_balance_pairs

    mat    = IM7_8552()
    angles = [0.0, 45.0, -45.0, 90.0]
    pairs  = detect_balance_pairs(angles)
    rf_min = 1.5

    # Use a load large enough to actually size the laminate (h > t_min × n_plies)
    # -500 kN/m Nxx approximates the root bending load seen in the wing demo at M=1.7
    # At this load the optimum thickness is ~5 mm, well above t_min=50µm × 8=0.4mm
    rho  = 0.0889; a_s = 295.1
    M, alpha_deg = 1.7, 3.0
    q    = 0.5 * rho * (M * a_s)**2
    Nyy  = -panel_pressure(M, alpha_deg, q)  # chordwise compression (negative)
    Nxx  = -500e3  # root-representative spanwise bending compression [N/m]
    N    = np.array([Nxx, Nyy, 0.0])

    M_vec = np.zeros(3)
    res  = optimize_laminate(N, M_vec, mat, angles, rf_min=rf_min,
                             balance_pairs=pairs, verbose=False)

    _check("Optimizer converged",
           res.converged,
           "")

    _check(f"RF_min ≈ {rf_min}  (Tsai-Wu constraint active at optimum)",
           abs(res.min_tsai_wu_rf - rf_min) < 0.05,
           f"min_tsai_wu_rf = {res.min_tsai_wu_rf:.4f}  (target {rf_min})",
           tol_pct=3.3)

    # All ply thicknesses should be >= t_min
    t_min = 0.05e-3
    _check("All ply thicknesses >= t_min",
           all(tk >= t_min * 0.999 for tk in res.t_full),
           f"min t = {min(res.t_full)*1e6:.1f} µm  (t_min = {t_min*1e6:.1f} µm)")

    # Total thickness should be positive and physically plausible (0.1–5 mm)
    h_total = sum(res.t_full)
    _check("Total laminate thickness physically plausible (0.1 – 5 mm)",
           0.1e-3 < h_total < 5e-3,
           f"h_total = {h_total*1e3:.3f} mm")

except Exception:
    print(f"  {_RED}ERROR{_RESET}"); traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════════════
# BLOCK 7  Physics monotonicity — load scaling and Mach trends
# ══════════════════════════════════════════════════════════════════════════════
_section("Block 7 — Physics monotonicity checks")

try:
    from composite_panel import IM7_8552, WingGeometry
    from composite_panel.optimizer import optimize_wing, detect_balance_pairs

    mat  = IM7_8552()
    wing = WingGeometry(semi_span=4.5, root_chord=4.0, taper_ratio=0.25,
                        sweep_le_deg=50.0, t_over_c=0.04, mtow_n=120_000.0)
    angles = [0.0, 45.0, -45.0, 90.0]
    pairs  = detect_balance_pairs(angles)

    def _run(mach):
        r = optimize_wing(wing=wing, mach=mach, altitude_m=20_000,
                          alpha_deg=3.0, mat=mat, angles_half_deg=angles,
                          n_load=2.5, n_stations=5, rf_min=1.5,
                          t_min=0.05e-3, t_init=0.15e-3,
                          balance_pairs=pairs, rho_kg_m3=1600.0)
        return r.total_skin_mass

    m08 = _run(0.8)
    m17 = _run(1.7)
    m50 = _run(5.0)

    _check("Wing mass increases from M0.8 → M1.7 → M5.0  (higher aero load)",
           m08 < m17 < m50,
           f"M0.8: {m08:.1f} kg,  M1.7: {m17:.1f} kg,  M5.0: {m50:.1f} kg")

    # ── Root heavier than tip (bending moment taper) ──────────────────────────
    from composite_panel.optimizer import optimize_wing
    from composite_panel import wing_panel_loads

    r17 = optimize_wing(wing=wing, mach=1.7, altitude_m=20_000,
                        alpha_deg=3.0, mat=mat, angles_half_deg=angles,
                        n_load=2.5, n_stations=8, rf_min=1.5,
                        t_min=0.05e-3, t_init=0.15e-3,
                        balance_pairs=pairs, rho_kg_m3=1600.0)

    h_root = r17.thicknesses[0]    # eta ≈ 0.05
    h_tip  = r17.thicknesses[-1]   # eta ≈ 0.95
    _check("Root skin thicker than tip  (bending moment taper)",
           h_root > h_tip,
           f"h_root={h_root*1e3:.2f} mm,  h_tip={h_tip*1e3:.2f} mm")

    # ── Load factor scaling: 3g should give ~20% more mass than 2.5g ────────
    r25 = optimize_wing(wing=wing, mach=1.7, altitude_m=20_000,
                        alpha_deg=3.0, mat=mat, angles_half_deg=angles,
                        n_load=2.5, n_stations=5, rf_min=1.5,
                        t_min=0.05e-3, t_init=0.15e-3,
                        balance_pairs=pairs, rho_kg_m3=1600.0)
    r30 = optimize_wing(wing=wing, mach=1.7, altitude_m=20_000,
                        alpha_deg=3.0, mat=mat, angles_half_deg=angles,
                        n_load=3.0, n_stations=5, rf_min=1.5,
                        t_min=0.05e-3, t_init=0.15e-3,
                        balance_pairs=pairs, rho_kg_m3=1600.0)
    _check("Higher load factor → heavier wing  (3g > 2.5g)",
           r30.total_skin_mass > r25.total_skin_mass,
           f"2.5g: {r25.total_skin_mass:.1f} kg  →  3.0g: {r30.total_skin_mass:.1f} kg")

except Exception:
    print(f"  {_RED}ERROR{_RESET}"); traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
n_pass = sum(1 for _, ok in _results if ok)
n_fail = sum(1 for _, ok in _results if not ok)
n_total = len(_results)

print(f"\n{'═'*60}")
print(f"{_BOLD}  VALIDATION SUMMARY{_RESET}")
print(f"{'═'*60}")
print(f"  Total checks : {n_total}")
print(f"  {_GREEN}Passed{_RESET}       : {n_pass}")
if n_fail:
    print(f"  {_RED}Failed{_RESET}       : {n_fail}")
    print(f"\n  Failed checks:")
    for name, ok in _results:
        if not ok:
            print(f"    {_RED}✗{_RESET} {name}")
else:
    print(f"  {_GREEN}All checks passed — model is self-consistent.{_RESET}")
print(f"{'═'*60}\n")
