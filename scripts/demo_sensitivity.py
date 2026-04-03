"""
demo_sensitivity.py
====================
Design sensitivity of optimal laminate mass to key flight parameters.

Shows how the minimum-mass laminate responds to changes in:
  1. Load factor n_load  (1.5 g → 4.0 g)
  2. Mach number         (1.4   → 4.0  )

For each sweep:
  - Optimal areal density [kg/m²] vs. parameter
  - Ply thickness breakdown — which angle groups drive the mass increase
  - Governing failure mode at each design point

MDO RELEVANCE
-------------
Each point on these curves is a full IPOPT solve with CasADi gradient
machinery.  The slope of the curve at any design point — the parametric
sensitivity d(m*)/dn_load — is available exactly from the KKT conditions in
a SINGLE solve without a second finite-difference run.

For an n-parameter MDO problem, this means:
  • Finite differences: N+1 solves per design point
  • CasADi parametric sensitivity: 1 solve, N sensitivities for free

The curves here are built from multiple solves for visualisation clarity.
In a production MDO loop, each solve also returns the full Jacobian.

Run:  python -X utf8 demo_sensitivity.py
"""

import sys, os, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from composite_panel import (
    IM7_8552, WingGeometry, wing_panel_loads,
    optimize_laminate, detect_balance_pairs,
)

# ── Palette ───────────────────────────────────────────────────────────────────
BG    = "#0a0e1a"; WHITE = "#e8edf5"; DIM = "#3a4060"
C_N   = "#4488ff"   # n_load sweep
C_M   = "#ff4455"   # Mach sweep
GOLD  = "#f0a030"
PLY_COLS = ["#00ddbb", "#ff8833", "#cc44ff", "#4488ff"]   # 0°, ±45°, 90°

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

# ── Configuration ─────────────────────────────────────────────────────────────
wing = WingGeometry(
    semi_span    = 4.5,
    root_chord   = 4.0,
    taper_ratio  = 0.25,
    sweep_le_deg = 50.0,
    t_over_c     = 0.04,
    mtow_n       = 120_000.0,
)

mat    = IM7_8552()
ALT_M  = 25_900.0
ALPHA  = 3.0
ETA    = 0.40          # representative mid-span station
RF_MIN = 1.5
PANEL_A, PANEL_B = 0.50, 0.20

ANGLES = [0.0, 45.0, -45.0, 90.0]
PAIRS  = detect_balance_pairs(ANGLES)

# Sweep ranges
N_VALS = np.array([1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
M_VALS = np.array([1.4, 1.8, 2.4, 3.0, 3.5, 4.0])
MACH_REF = 2.4     # fixed Mach for n_load sweep
N_REF    = 2.5     # fixed n_load for Mach sweep

# ── n_load sweep ─────────────────────────────────────────────────────────────
print("=" * 65)
print("  Sensitivity Sweep 1: optimal mass vs. load factor")
print("=" * 65)

mass_n  = []
t_half_n = []
rf_n    = []

for n_val in N_VALS:
    pl = wing_panel_loads(wing, ETA, MACH_REF, ALT_M, ALPHA, float(n_val))
    r  = optimize_laminate(
        N_loads=pl.N, M_loads=pl.M, mat=mat,
        angles_half_deg=ANGLES, rf_min=RF_MIN,
        balance_pairs=PAIRS,
        panel_a=PANEL_A, panel_b=PANEL_B,
        verbose=False,
    )
    mass_n.append(r.areal_density)
    t_half_n.append(r.t_half * 1e3)   # mm
    rf_n.append(r.min_tsai_wu_rf)
    print(f"  n={n_val:.1f}g  →  rho·h={r.areal_density:.4f} kg/m²  "
          f"RF={r.min_tsai_wu_rf:.3f}  conv={r.converged}")

mass_n   = np.array(mass_n)
t_half_n = np.array(t_half_n)   # (n_n, n_half)

# Numerical gradient (central differences where available)
dm_dn = np.gradient(mass_n, N_VALS)
print(f"\n  d(m*)/d(n_load) at n=2.5g  ≈  {dm_dn[2]:.4f} kg/m² per g")

# ── Mach sweep ────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  Sensitivity Sweep 2: optimal mass vs. Mach")
print("=" * 65)

mass_m   = []
t_half_m = []
rf_m     = []

for m_val in M_VALS:
    pl = wing_panel_loads(wing, ETA, float(m_val), ALT_M, ALPHA, N_REF)
    r  = optimize_laminate(
        N_loads=pl.N, M_loads=pl.M, mat=mat,
        angles_half_deg=ANGLES, rf_min=RF_MIN,
        balance_pairs=PAIRS,
        panel_a=PANEL_A, panel_b=PANEL_B,
        verbose=False,
    )
    mass_m.append(r.areal_density)
    t_half_m.append(r.t_half * 1e3)
    rf_m.append(r.min_tsai_wu_rf)
    print(f"  M={m_val:.1f}  →  rho·h={r.areal_density:.4f} kg/m²  "
          f"RF={r.min_tsai_wu_rf:.3f}  conv={r.converged}")

mass_m   = np.array(mass_m)
t_half_m = np.array(t_half_m)

dm_dM = np.gradient(mass_m, M_VALS)
print(f"\n  d(m*)/d(Mach) at M=2.4  ≈  {dm_dM[2]:.4f} kg/m² per Mach unit")

# ── Tornado chart: fractional sensitivity ─────────────────────────────────────
# Each sensitivity is normalised: (delta_mass / mass_nominal) / (delta_param / param_nominal)
mass_nom_n = np.interp(N_REF, N_VALS, mass_n)
mass_nom_m = np.interp(MACH_REF, M_VALS, mass_m)
dm_dn_norm = np.interp(N_REF, N_VALS, dm_dn) * N_REF / mass_nom_n    # d(lnm)/d(lnn)
dm_dM_norm = np.interp(MACH_REF, M_VALS, dm_dM) * MACH_REF / mass_nom_m

print(f"\n  Normalised sensitivities at nominal design point:")
print(f"    ∂(ln m*)/∂(ln n_load)  =  {dm_dn_norm:.3f}")
print(f"    ∂(ln m*)/∂(ln Mach)    =  {dm_dM_norm:.3f}")
print(f"  (1.0 = doubling the parameter doubles the mass)")

# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 9), facecolor=BG)
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.48, wspace=0.35,
                        left=0.07, right=0.97, top=0.87, bottom=0.09)
# Bottom row uses only 2 of 3 columns; tornado spans both remaining columns
gs_bot = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1, :],
                                          wspace=0.35)

# (A) Mass vs n_load
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(N_VALS, mass_n, color=C_N, lw=2.5, marker="o", markersize=6)
# Annotate gradient at nominal point
idx_n = np.argmin(np.abs(N_VALS - N_REF))
ax1.annotate(
    f"slope ≈ {dm_dn[idx_n]:+.4f} kg/m²/g",
    xy=(N_VALS[idx_n], mass_n[idx_n]),
    xytext=(N_VALS[idx_n] + 0.3, mass_n[idx_n] - 0.02),
    color=GOLD, fontsize=7,
    arrowprops=dict(arrowstyle="->", color=GOLD, lw=0.8),
)
ax1.set_xlabel("Load factor  n  [g]", fontsize=9)
ax1.set_ylabel("Optimal areal density  [kg/m²]", fontsize=9)
ax1.set_title("Mass vs load factor", fontsize=10, pad=6)
_style(ax1)

# (B) Mass vs Mach
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(M_VALS, mass_m, color=C_M, lw=2.5, marker="o", markersize=6)
idx_m = np.argmin(np.abs(M_VALS - MACH_REF))
ax2.annotate(
    f"slope ≈ {dm_dM[idx_m]:+.4f} kg/m²/Mach",
    xy=(M_VALS[idx_m], mass_m[idx_m]),
    xytext=(M_VALS[idx_m] + 0.2, mass_m[idx_m] - 0.02),
    color=GOLD, fontsize=7,
    arrowprops=dict(arrowstyle="->", color=GOLD, lw=0.8),
)
ax2.set_xlabel("Mach  [-]", fontsize=9)
ax2.set_ylabel("Optimal areal density  [kg/m²]", fontsize=9)
ax2.set_title("Mass vs Mach", fontsize=10, pad=6)
_style(ax2)

# (C) Tornado chart
ax3 = fig.add_subplot(gs[0, 2])
labels_t = ["∂lnm/∂ln n_load", "∂lnm/∂ln Mach"]
vals_t   = [dm_dn_norm, dm_dM_norm]
cols_t   = [C_N, C_M]
y_pos = np.arange(len(labels_t))
ax3.barh(y_pos, vals_t, color=cols_t, edgecolor=DIM, alpha=0.85)
ax3.axvline(0, color=DIM, lw=0.6)
ax3.set_yticks(y_pos)
ax3.set_yticklabels(labels_t, fontsize=8)
ax3.set_xlabel("Normalised sensitivity  d(ln m*)/d(ln p)", fontsize=9)
ax3.set_title("Logarithmic sensitivity\n(1 = proportional scaling)", fontsize=10, pad=6)
_style(ax3)

# (D) Ply breakdown vs n_load (stacked area)
ax4 = fig.add_subplot(gs_bot[0])
t0   = t_half_n[:, 0]
t45  = t_half_n[:, 1] + t_half_n[:, 2]
t90  = t_half_n[:, 3]

ax4.stackplot(N_VALS, t90, t45, t0,
              labels=["90° plies", "±45° plies", "0° plies"],
              colors=[PLY_COLS[3], PLY_COLS[1], PLY_COLS[0]], alpha=0.85)
ax4.set_xlabel("Load factor  n  [g]", fontsize=9)
ax4.set_ylabel("Half-stack thickness  [mm]", fontsize=9)
ax4.set_title("Ply breakdown vs load factor", fontsize=10, pad=6)
_style(ax4, legend=True)

# (E) Ply breakdown vs Mach
ax5 = fig.add_subplot(gs_bot[1])
t0_m  = t_half_m[:, 0]
t45_m = t_half_m[:, 1] + t_half_m[:, 2]
t90_m = t_half_m[:, 3]

ax5.stackplot(M_VALS, t90_m, t45_m, t0_m,
              labels=["90° plies", "±45° plies", "0° plies"],
              colors=[PLY_COLS[3], PLY_COLS[1], PLY_COLS[0]], alpha=0.85)
ax5.set_xlabel("Mach  [-]", fontsize=9)
ax5.set_ylabel("Half-stack thickness  [mm]", fontsize=9)
ax5.set_title("Ply breakdown vs Mach", fontsize=10, pad=6)
_style(ax5, legend=True)

# ── Header ────────────────────────────────────────────────────────────────────
fig.text(
    0.5, 0.955,
    f"Laminate Sensitivity Analysis  |  IM7/8552  [0/±45/90]s  |  "
    f"η={ETA}  Alt={ALT_M/1e3:.0f} km",
    ha="center", color=WHITE, fontsize=12, fontweight="bold",
)
fig.text(
    0.5, 0.929,
    "Each point = 1 IPOPT solve.  Slope at any design point is the CasADi "
    "parametric sensitivity d(m*)/d(p) — available analytically from KKT "
    "conditions without a second solve.",
    ha="center", color=DIM, fontsize=8,
)

os.makedirs("outputs", exist_ok=True)
outpath = "outputs/sensitivity_analysis.png"
plt.savefig(outpath, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"\n  Figure saved → {outpath}")
plt.close()
