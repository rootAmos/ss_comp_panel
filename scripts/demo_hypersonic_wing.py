"""
demo_hypersonic_wing.py
=======================
Wing-skin sizing across the full flight envelope: M0.8 to M5.

Compares four Mach numbers (subsonic → low-supersonic → moderate-supersonic →
hypersonic) plus a thermal+buckling case at Mach 5.

Run:  python -X utf8 demo_hypersonic_wing.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as _np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

from composite_panel import (
    IM7_8552, IM7_8552_thermal,
    WingGeometry, wing_panel_loads,
    aero_wall_temperature, equilibrium_wall_temperature,
    thermal_state_from_flight, thermal_resultants,
    Ply, Laminate,
    Nxx_cr, buckling_rf,
)
from composite_panel.optimizer import (
    optimize_wing, detect_balance_pairs,
    WingOptimizationResult,
)

# ── Palette ──────────────────────────────────────────────────────────────────
BG    = "#0a0e1a"; WHITE = "#e8edf5"; DIM   = "#3a4060"
C_08  = "#4488ff"   # M0.8  — blue
C_17  = "#00ddbb"   # M1.7  — teal
C_24  = "#ff8833"   # M2.4  — orange
C_50  = "#ff4455"   # M5.0  — red
C_5T  = "#cc44ff"   # M5.0 therm+buckle — violet
GOLD  = "#f0a030"

COLORS = [C_08, C_17, C_24, C_50, C_5T]

def _style(ax, legend=False):
    ax.set_facecolor(BG)
    ax.tick_params(colors=WHITE, labelsize=8)
    ax.xaxis.label.set_color(WHITE)
    ax.yaxis.label.set_color(WHITE)
    ax.title.set_color(WHITE)
    for spine in ax.spines.values():
        spine.set_edgecolor(DIM)
    ax.grid(color=DIM, linewidth=0.4, alpha=0.5)
    if legend:
        ax.legend(fontsize=7, framealpha=0.15, labelcolor=WHITE)

# ── Wing definition ───────────────────────────────────────────────────────────
wing = WingGeometry(
    semi_span    = 4.5,
    root_chord   = 4.0,
    taper_ratio  = 0.25,
    sweep_le_deg = 50.0,
    t_over_c     = 0.04,
    mtow_n       = 120_000.0,
)

mat = IM7_8552()
pt  = IM7_8552_thermal()

ANGLES     = [0.0, 45.0, -45.0, 90.0]
PAIRS      = detect_balance_pairs(ANGLES)
PANEL_A    = 0.50
PANEL_B    = 0.20
ETAS       = _np.linspace(0.05, 0.95, 12)
N_STATIONS = len(ETAS)
ALT_M      = 25_900   # SR-71 Blackbird cruise altitude (85,000 ft)
ALPHA_DEG  = 3.0
N_LOAD     = 2.5

# ── Cases ─────────────────────────────────────────────────────────────────────
cases = {
    "M0.8  subsonic":      dict(mach=0.8, use_thermal=False, use_buckling=False, color=C_08),
    "M1.7  supersonic":    dict(mach=1.7, use_thermal=False, use_buckling=False, color=C_17),
    "M2.4  supersonic":    dict(mach=2.4, use_thermal=False, use_buckling=False, color=C_24),
    "M5.0  mech only":     dict(mach=5.0, use_thermal=False, use_buckling=False, color=C_50),
    "M5.0  therm+buckle":  dict(mach=5.0, use_thermal=True,  use_buckling=True,  color=C_5T),
}

print("=" * 65)
print("  Wing Skin Sizing  |  IM7/8552  [0/+/-45/90]s  |  M0.8->M5")
print("=" * 65)

results_all = {}
load_arrays = {}

for label, cfg in cases.items():
    mach         = cfg["mach"]
    use_thermal  = cfg["use_thermal"]
    use_buckling = cfg["use_buckling"]

    print(f"\n{label} ...")

    ts_list = []
    for eta in ETAS:
        if use_thermal:
            x_c = 0.4 * wing.chord(eta)
            ts_list.append(thermal_state_from_flight(mach, ALT_M, x_c))
        else:
            ts_list.append(None)

    result = optimize_wing(
        wing            = wing,
        mach            = mach,
        altitude_m      = ALT_M,
        alpha_deg       = ALPHA_DEG,
        mat             = mat,
        angles_half_deg = ANGLES,
        n_load          = N_LOAD,
        n_stations      = N_STATIONS,
        rf_min          = 1.5,
        t_min           = 0.05e-3,
        t_init          = 0.15e-3,
        balance_pairs   = PAIRS,
        rho_kg_m3       = 1600.0,
        panel_a         = PANEL_A if use_buckling else None,
        panel_b         = PANEL_B if use_buckling else None,
        buckle_rf_min   = 1.0,
        thermal_states  = ts_list if use_thermal else None,
        ply_thermal     = pt if use_thermal else None,
    )

    results_all[label] = result

    Nxx_arr, Nyy_arr, NT_arr, buckle_arr = [], [], [], []
    for i, eta in enumerate(ETAS):
        pl    = result.loads[i]
        n_ply = 2 * len(ANGLES)
        ang_full = ANGLES + list(reversed(ANGLES))
        lam_i = Laminate([Ply(mat, result.opt_results[i].t_full[k], ang_full[k])
                          for k in range(n_ply)])
        Nxx_arr.append(pl.Nxx / 1e3)
        Nyy_arr.append(pl.Nyy / 1e3)

        if use_thermal and ts_list[i] is not None:
            NT, _ = thermal_resultants(lam_i.plies, [pt]*n_ply, ts_list[i],
                                       lam_i.z_interfaces)
            NT_arr.append(NT[0] / 1e3)
        else:
            NT_arr.append(0.0)

        RF_b = buckling_rf(pl.N, lam_i.D, PANEL_A, PANEL_B)
        buckle_arr.append(RF_b)

    load_arrays[label] = dict(
        Nxx=_np.array(Nxx_arr), Nyy=_np.array(Nyy_arr),
        NT=_np.array(NT_arr),   buckle_rf=_np.array(buckle_arr),
    )

# ── Figure — 2x2 grid ─────────────────────────────────────────────────────────
fig = plt.figure(figsize=(13, 9), facecolor=BG)
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.46, wspace=0.32,
                        left=0.08, right=0.97, top=0.88, bottom=0.08)

# (1) Spanwise thickness
ax1 = fig.add_subplot(gs[0, 0])
for label, cfg in cases.items():
    wr = results_all[label]
    ax1.plot(ETAS, wr.thicknesses * 1e3, color=cfg["color"], lw=2.0,
             marker="o", markersize=3, label=label)
ax1.set_xlabel("eta = y/b  [-]", fontsize=9)
ax1.set_ylabel("Total laminate h  [mm]", fontsize=9)
ax1.set_title("Spanwise thickness", fontsize=10, pad=6)
_style(ax1, legend=True)

# (2) Spanwise Nxx (primary bending driver)
ax2 = fig.add_subplot(gs[0, 1])
for label, cfg in cases.items():
    la = load_arrays[label]
    ax2.plot(ETAS, la["Nxx"], color=cfg["color"], lw=2.0, label=label)
ax2.axhline(0, color=DIM, lw=0.5)
ax2.set_xlabel("eta = y/b  [-]", fontsize=9)
ax2.set_ylabel("Nxx  [kN/m]", fontsize=9)
ax2.set_title("Spanwise compression (bending)", fontsize=10, pad=6)
_style(ax2, legend=True)

# (3) Skin mass bar
ax4 = fig.add_subplot(gs[1, 0])
labels_bar = list(cases.keys())
masses_bar = [results_all[l].total_skin_mass for l in labels_bar]
colors_bar = [cfg["color"] for cfg in cases.values()]
bars = ax4.barh(labels_bar, masses_bar, color=colors_bar, edgecolor=DIM, alpha=0.85)
ax4.bar_label(bars, fmt="%.1f kg", color=WHITE, fontsize=8, padding=4)
ax4.set_xlabel("Semi-span upper-skin mass [kg]", fontsize=9)
ax4.set_title("Total skin mass", fontsize=10, pad=6)
_style(ax4)

# (4) Areal density penalty vs M0.8 baseline
ax5 = fig.add_subplot(gs[1, 1])
rho_base = results_all["M0.8  subsonic"].areal_densities
for label, cfg in cases.items():
    wr  = results_all[label]
    pct = (wr.areal_densities - rho_base) / rho_base * 100
    ax5.plot(ETAS, pct, color=cfg["color"], lw=2.0, label=label)
ax5.axhline(0, color=C_08, lw=1.0, linestyle="--", alpha=0.5)
ax5.set_xlabel("eta = y/b  [-]", fontsize=9)
ax5.set_ylabel("Mass penalty vs M0.8  [%]", fontsize=9)
ax5.set_title("Areal density penalty", fontsize=10, pad=6)
_style(ax5, legend=True)

# ── Header ────────────────────────────────────────────────────────────────────
fig.text(
    0.5, 0.955,
    f"Wing Skin MDO  |  {mat.name}  [0/+/-45/90]s  "
    f"|  b/2={wing.semi_span}m  c_root={wing.root_chord}m  sweep={wing.sweep_le_deg}deg",
    ha="center", color=WHITE, fontsize=12, fontweight="bold",
)
fig.text(
    0.5, 0.928,
    f"Alt = {ALT_M/1e3:.0f} km  |  alpha = {ALPHA_DEG}deg  |  n = {N_LOAD}g  "
    f"|  Panel {PANEL_A*100:.0f}x{PANEL_B*100:.0f}cm  |  RF_TsaiWu >= 1.5  |  RF_buckle >= 1.0",
    ha="center", color=DIM, fontsize=8,
)

outpath = "outputs/hypersonic_wing_mdo.png"
plt.savefig(outpath, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"\n  Figure saved -> {outpath}")

print("\n  Semi-span upper-skin mass:")
m_base = results_all["M0.8  subsonic"].total_skin_mass
for lbl in cases:
    m = results_all[lbl].total_skin_mass
    pct = (m - m_base) / m_base * 100
    sign = "+" if pct >= 0 else ""
    print(f"    {lbl:<28}  {m:6.1f} kg  ({sign}{pct:.0f}% vs M0.8)")

plt.close()
