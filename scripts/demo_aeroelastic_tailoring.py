"""
demo_aeroelastic_tailoring.py
==============================
Simultaneous strength + aeroelastic tailoring via a single CasADi optimisation.

THREE CASES
-----------
1. STRENGTH ONLY (balanced [0/±45/90]s)
   Standard Tsai-Wu + buckling constraints.  No aeroelastic awareness.
   Sets the baseline mass.

2. GEOMETRY WASHOUT — balanced, fixed angles
   Same balanced stack, but an aeroelastic constraint is added:
       Δα_tip ≤ −relief_min_deg
   The optimizer must find ply thicknesses that are simultaneously strong
   AND compliant enough for sweep-induced washout.  D16 = 0 (balanced).

3. BEND-TWIST COUPLED — unbalanced, free angles
   Angles and thicknesses are both design variables.  The unbalanced stack
   allows D16 ≠ 0, giving a second lever: bend-twist coupling augments the
   sweep-induced washout so the compliance constraint is met at lower mass
   penalty.

WHY THIS MATTERS
----------------
The CasADi / IPOPT solve treats the aeroelastic constraint the same as any
other: exact sparse Jacobians flow through EI → A11 → ply thicknesses and
through D16, D66 → ply angles.  There is no inner loop, no finite-difference
perturbation, and no sequentially coupled solve.  The multi-physics problem
(structural strength + aeroelastic compliance) is solved in one shot.

Run:  python -X utf8 demo_aeroelastic_tailoring.py
"""

import sys, os, math, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from composite_panel import (
    IM7_8552, WingGeometry,
    Ply, Laminate,
    optimize_laminate, optimize_laminate_aeroelastic, detect_balance_pairs,
    static_aeroelastic,
)

# ── Palette ───────────────────────────────────────────────────────────────────
BG    = "#0a0e1a"; WHITE = "#e8edf5"; DIM = "#3a4060"
C1    = "#4488ff"   # strength only
C2    = "#ff8833"   # geometry washout (balanced)
C3    = "#00ddbb"   # bend-twist coupled (unbalanced)
GOLD  = "#f0a030"

def _style(ax, legend=False):
    ax.set_facecolor(BG)
    ax.tick_params(colors=WHITE, labelsize=8)
    ax.xaxis.label.set_color(WHITE)
    ax.yaxis.label.set_color(WHITE)
    ax.title.set_color(WHITE)
    for sp in ax.spines.values():
        sp.set_edgecolor(DIM)
    ax.grid(color=DIM, linewidth=0.4, alpha=0.5)
    if legend:
        ax.legend(fontsize=7, framealpha=0.15, labelcolor=WHITE,
                  facecolor=BG, edgecolor=DIM)

# ── Wing + material ───────────────────────────────────────────────────────────
# Use a smaller, more flexible wing where aeroelastic effects are meaningful.
# Root chord 2 m, thin section, 45° sweep — aeroelastic relief is ~0.1° at
# this scale, so setting relief_min_deg = 0.05° makes the constraint active.
wing = WingGeometry(
    semi_span    = 4.5,
    root_chord   = 2.0,
    taper_ratio  = 0.30,
    sweep_le_deg = 45.0,
    t_over_c     = 0.04,
    mtow_n       = 60_000.0,
)

mat    = IM7_8552()
MACH   = 1.7
ALT_M  = 15_000.0
ALPHA  = 3.5
N_LOAD = 2.5

# Loads at a representative mid-span station (eta=0.4)
from composite_panel import wing_panel_loads
_pl = wing_panel_loads(wing, 0.40, MACH, ALT_M, ALPHA, N_LOAD)
N_LOADS = _pl.N          # [Nxx, Nyy, Nxy]
M_LOADS = _pl.M          # [Mxx, Myy, Mxy]

RF_MIN          = 1.5
RELIEF_MIN_DEG  = 0.05    # target washout [deg] — makes BT coupling measurable

ANGLES     = [0.0, 45.0, -45.0, 90.0]
PAIRS      = detect_balance_pairs(ANGLES)
PANEL_A    = 0.30
PANEL_B    = 0.10

print("=" * 65)
print("  Aeroelastic Tailoring  |  IM7/8552  |  M1.7  |  3 Cases")
print("=" * 65)
print(f"  Loads at eta=0.40: Nxx={N_LOADS[0]/1e3:.1f} kN/m  "
      f"Nyy={N_LOADS[1]/1e3:.1f} kN/m  Nxy={N_LOADS[2]/1e3:.1f} kN/m")
print(f"  Relief min  : {RELIEF_MIN_DEG}°  (tip washout must be >= {RELIEF_MIN_DEG}°)\n")

# ── Case 1: Strength only ─────────────────────────────────────────────────────
print("Case 1: Strength only (balanced, no aeroelastic constraint) ...")
r1 = optimize_laminate(
    N_loads=N_LOADS, M_loads=M_LOADS, mat=mat,
    angles_half_deg=ANGLES, rf_min=RF_MIN,
    balance_pairs=PAIRS,
    panel_a=PANEL_A, panel_b=PANEL_B,
    verbose=False,
)
print(f"  mass proxy (rho·h): {r1.areal_density:.4f} kg/m²  RF={r1.min_tsai_wu_rf:.3f}  "
      f"conv={r1.converged}")

# Post-compute washout for this design
_lam1 = Laminate([Ply(mat, r1.t_full[k], (r1.angles_half + list(reversed(r1.angles_half)))[k])
                  for k in range(len(r1.t_full))])
_ae1 = static_aeroelastic(
    wing=wing, mach=MACH, altitude_m=ALT_M,
    alpha_rigid_deg=ALPHA, n_load=N_LOAD,
    laminate_A11=float(_lam1.A[0, 0]),
    laminate_h=r1.total_h,
    laminate_D16=float(_lam1.D[0, 2]),
    laminate_D66=float(_lam1.D[2, 2]),
)
washout1 = _ae1.delta_alpha[-1]
print(f"  Tip washout (post-check): {washout1:+.4f}°")

# ── Case 2: Geometry washout (balanced) ───────────────────────────────────────
print("\nCase 2: Geometry washout — balanced angles, EI constraint ...")
r2 = optimize_laminate_aeroelastic(
    N_loads=N_LOADS, M_loads=M_LOADS, mat=mat,
    angles_half_deg=ANGLES,
    wing=wing, n_load=N_LOAD, relief_min_deg=RELIEF_MIN_DEG,
    use_bt_coupling=False,          # D16 = 0, rely on EI compliance only
    rf_min=RF_MIN,
    panel_a=PANEL_A, panel_b=PANEL_B,
    verbose=False,
)
print(f"  mass proxy (rho·h): {r2.areal_density:.4f} kg/m²  RF={r2.min_tsai_wu_rf:.3f}  "
      f"conv={r2.converged}")
print(f"  Washout achieved (CasADi): {r2.achieved_washout_deg:+.4f}°  "
      f"(target <= {-RELIEF_MIN_DEG:.3f}°)")
print(f"  EI root: {r2.EI_root:.3e} N·m²   D16={r2.D16:.4f} N·m")

# ── Case 3: Bend-twist coupled ────────────────────────────────────────────────
print("\nCase 3: Bend-twist coupled — unbalanced angles, D16 ≠ 0 ...")
r3 = optimize_laminate_aeroelastic(
    N_loads=N_LOADS, M_loads=M_LOADS, mat=mat,
    angles_half_deg=ANGLES,
    wing=wing, n_load=N_LOAD, relief_min_deg=RELIEF_MIN_DEG,
    use_bt_coupling=True,           # angles free, D16 optimised
    rf_min=RF_MIN,
    panel_a=PANEL_A, panel_b=PANEL_B,
    verbose=False,
)
print(f"  mass proxy (rho·h): {r3.areal_density:.4f} kg/m²  RF={r3.min_tsai_wu_rf:.3f}  "
      f"conv={r3.converged}")
print(f"  Washout achieved (CasADi): {r3.achieved_washout_deg:+.4f}°  "
      f"(target <= {-RELIEF_MIN_DEG:.3f}°)")
print(f"  EI root: {r3.EI_root:.3e} N·m²   D16={r3.D16:.4f} N·m")
print(f"  EK/GJ (bt ratio): {r3.bt_ratio_root:.5f}")
print(f"  Optimal angles: {[f'{a:.1f}' for a in r3.angles_half]}°")

# Post-check case 3 with full static_aeroelastic
_lam3 = Laminate([Ply(mat, r3.t_full[k], (r3.angles_half + list(reversed(r3.angles_half)))[k])
                  for k in range(len(r3.t_full))])
_ae3 = static_aeroelastic(
    wing=wing, mach=MACH, altitude_m=ALT_M,
    alpha_rigid_deg=ALPHA, n_load=N_LOAD,
    laminate_A11=float(_lam3.A[0, 0]),
    laminate_h=r3.total_h,
    laminate_D16=float(_lam3.D[0, 2]),
    laminate_D66=float(_lam3.D[2, 2]),
)
washout3 = _ae3.delta_alpha[-1]
print(f"  Tip washout (post-check, full aeroelastic): {washout3:+.4f}°")

# ── Mass penalty analysis ─────────────────────────────────────────────────────
mass_base = r1.areal_density
penalty2  = (r2.areal_density - mass_base) / mass_base * 100 if r2.converged else float("nan")
penalty3  = (r3.areal_density - mass_base) / mass_base * 100 if r3.converged else float("nan")

print(f"\n  Mass summary:")
print(f"    Case 1 (strength only)       : {mass_base:.4f} kg/m²  baseline")
print(f"    Case 2 (geometry washout)    : {r2.areal_density:.4f} kg/m²  "
      f"({'+' if penalty2>=0 else ''}{penalty2:.1f}% vs baseline)")
print(f"    Case 3 (bend-twist coupled)  : {r3.areal_density:.4f} kg/m²  "
      f"({'+' if penalty3>=0 else ''}{penalty3:.1f}% vs baseline)")

# ── Figure ────────────────────────────────────────────────────────────────────
cases  = ["Strength\nonly", "Geometry\nwashout", "BT\ncoupled"]
masses = [r1.areal_density, r2.areal_density, r3.areal_density]
colors = [C1, C2, C3]
washouts = [washout1,
            r2.achieved_washout_deg if r2.converged else 0.0,
            r3.achieved_washout_deg if r3.converged else 0.0]
D16s   = [0.0, r2.D16, r3.D16]
EIs    = [None, r2.EI_root, r3.EI_root]

fig = plt.figure(figsize=(13, 8), facecolor=BG)
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.50, wspace=0.38,
                        left=0.07, right=0.97, top=0.87, bottom=0.09)

# (A) Areal density comparison
ax_mass = fig.add_subplot(gs[0, 0])
bars = ax_mass.bar(cases, masses, color=colors, edgecolor=DIM, width=0.55, alpha=0.88)
ax_mass.bar_label(bars, fmt="%.3f\nkg/m²", color=WHITE, fontsize=7, padding=3)
ax_mass.set_ylabel("Areal density  [kg/m²]", fontsize=9)
ax_mass.set_title("Mass comparison", fontsize=10, pad=6)
_style(ax_mass)

# (B) Tip washout comparison
ax_wo = fig.add_subplot(gs[0, 1])
wo_vals = [abs(w) for w in washouts]
bars_wo = ax_wo.bar(cases, wo_vals, color=colors, edgecolor=DIM, width=0.55, alpha=0.88)
ax_wo.axhline(RELIEF_MIN_DEG, color=GOLD, lw=1.2, linestyle="--", label=f"Target {RELIEF_MIN_DEG}°")
ax_wo.bar_label(bars_wo, fmt="%.4f°", color=WHITE, fontsize=7, padding=3)
ax_wo.set_ylabel("|Δα_tip|  [deg]  (washout magnitude)", fontsize=9)
ax_wo.set_title("Achieved washout", fontsize=10, pad=6)
_style(ax_wo, legend=True)

# (C) D16 comparison
ax_d16 = fig.add_subplot(gs[0, 2])
bars_d16 = ax_d16.bar(cases, D16s, color=colors, edgecolor=DIM, width=0.55, alpha=0.88)
ax_d16.bar_label(bars_d16, fmt="%.4f\nN·m", color=WHITE, fontsize=7, padding=3)
ax_d16.set_ylabel("D16  [N·m]  (bend-twist coupling)", fontsize=9)
ax_d16.set_title("Bend-twist coupling stiffness", fontsize=10, pad=6)
_style(ax_d16)

# (D) Spanwise effective-AoA from full static_aeroelastic post-check
ax_ae = fig.add_subplot(gs[1, :2])
ax_ae.plot(_ae1.etas, _ae1.delta_alpha, color=C1, lw=2.0, label="Case 1  strength only")
if r2.converged:
    _lam2 = Laminate([Ply(mat, r2.t_full[k],
                          (r2.angles_half + list(reversed(r2.angles_half)))[k])
                      for k in range(len(r2.t_full))])
    _ae2 = static_aeroelastic(
        wing=wing, mach=MACH, altitude_m=ALT_M,
        alpha_rigid_deg=ALPHA, n_load=N_LOAD,
        laminate_A11=float(_lam2.A[0, 0]),
        laminate_h=r2.total_h,
        laminate_D16=float(_lam2.D[0, 2]),
        laminate_D66=float(_lam2.D[2, 2]),
    )
    ax_ae.plot(_ae2.etas, _ae2.delta_alpha, color=C2, lw=2.0,
               label="Case 2  geometry washout")
ax_ae.plot(_ae3.etas, _ae3.delta_alpha, color=C3, lw=2.0, linestyle="--",
           label="Case 3  BT coupled")
ax_ae.axhline(-RELIEF_MIN_DEG, color=GOLD, lw=1.0, linestyle=":", alpha=0.7,
              label=f"Target Δα = {-RELIEF_MIN_DEG}°")
ax_ae.axhline(0, color=DIM, lw=0.5)
ax_ae.set_xlabel("η = y/b  [-]", fontsize=9)
ax_ae.set_ylabel("Δα_tip  [deg]  (washout < 0)", fontsize=9)
ax_ae.set_title("Spanwise twist relief — post-check with full static_aeroelastic()",
                fontsize=10, pad=6)
_style(ax_ae, legend=True)

# (E) Ply thickness breakdown case 3
ax_ply = fig.add_subplot(gs[1, 2])
n_half = len(ANGLES)
ang_full3 = r3.angles_half + list(reversed(r3.angles_half))
x_pos = np.arange(len(r3.t_full))
bar_cols = [C3 if abs(a) < 5 else (C1 if abs(a - 45) < 10 or abs(a + 45) < 10 else GOLD)
            for a in ang_full3]
ax_ply.bar(x_pos, r3.t_full * 1e3, color=bar_cols, edgecolor=DIM, alpha=0.85)
ax_ply.set_xticks(x_pos)
ax_ply.set_xticklabels([f"{a:.0f}°" for a in ang_full3], fontsize=7, rotation=45)
ax_ply.set_ylabel("Ply thickness  [mm]", fontsize=9)
ax_ply.set_title("Case 3: Ply stack at optimum", fontsize=10, pad=6)
_style(ax_ply)

# ── Header ────────────────────────────────────────────────────────────────────
fig.text(
    0.5, 0.957,
    "Aeroelastic Tailoring Optimiser  |  IM7/8552  |  M1.7  @  15 km  |  n = 2.5g",
    ha="center", color=WHITE, fontsize=12, fontweight="bold",
)
fig.text(
    0.5, 0.931,
    f"Washout target: Δα_tip ≤ −{RELIEF_MIN_DEG}°  |  "
    f"RF_TsaiWu ≥ {RF_MIN}  |  "
    f"Aeroelastic constraint CasADi-native (no inner loop, exact Jacobians)",
    ha="center", color=DIM, fontsize=8,
)

os.makedirs("outputs", exist_ok=True)
outpath = "outputs/aeroelastic_tailoring.png"
plt.savefig(outpath, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"\n  Figure saved → {outpath}")
plt.close()
