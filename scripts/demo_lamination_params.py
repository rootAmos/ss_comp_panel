"""
demo_lamination_params.py
=========================
Lamination parameters, stiffness polars, and aeroelastic tailoring
for composite wing skin panels.

OVERVIEW
--------
This demo covers the three features suggested by aerostructures colleagues:

  1. STIFFNESS POLARS
     Plot in-plane (Ex, Ey, Gxy) and out-of-plane (D11, D22, D66, D16, D26)
     stiffness as a function of single-angle fibre orientation θ.
     Immediately reveals which angles maximise specific stiffness components
     without running an optimisation.

  2. LAMINATION PARAMETER SPACE
     Compute lamination parameters (LPs) for several common layup families
     ([0/90]_s, [±45]_s, QI, off-axis) and plot them in the ξ1A–ξ2A feasibility
     domain.  LPs are angle-independent stiffness coordinates that span a convex
     domain, making them ideal for gradient-based optimisation.

  3. AEROELASTIC TAILORING WITH UNBALANCED LAMINATES
     Compare the aeroelastic response of a swept wing skin laminated with:
       (a) Balanced [±θ/0/90]_s  (D16 = D26 = 0)
       (b) Unbalanced [θ/0/90]_s  (D16 ≠ 0 → bend-twist coupling)
     The unbalanced laminate engineers passive wash-out — the wing twist
     relieves lift loads under bending without any actuator.

PHYSICS BACKGROUND
------------------
For an unbalanced laminate, bending curvature κx induces torsional rotation
φ through the coupling stiffness EK = 2 · D16 · b_box:

    φ(y) = −(EK/GJ) · θ_bending(y)

On a swept-back wing, bending → nose-down twist (wash-out) already from
geometry.  An unbalanced laminate with positive D16 augments this effect,
reducing the effective angle of attack tip-inward and thus relieving span-wise
bending loads under high-g manoeuvres.

References
----------
Gürdal, Z., Haftka, R.T. & Hajela, P. — Design and Optimization of Laminated
    Composite Materials (Wiley, 1999), Ch. 4
Weisshaar, T.A. — Aeroelastic tailoring of forward swept composite wings
    (J. Aircraft, 1981, 18(8), pp. 669-676)
Miki, M. — Material design of composite laminates with required in-plane
    elastic properties (1982) ICCM-4, pp. 1725-1731.

Run:  python demo_lamination_params.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

from composite_panel import Ply, Laminate, IM7_8552, WingGeometry
from composite_panel import static_aeroelastic
from composite_panel.lamination_parameters import (
    material_invariants,
    lamination_parameters,
    abd_from_lamination_params,
    stiffness_polar,
    plot_lp_feasibility,
    bend_twist_coupling_index,
    shear_extension_coupling_index,
)

# ─── Colour palette (dark theme consistent with other demos) ─────────────────
BG     = '#0d1117'
PANEL  = '#161b22'
BORDER = '#30363d'
TEXT   = '#c9d1d9'
TITLE  = '#f0f6fc'
BLUE   = '#58a6ff'
RED    = '#f78166'
GREEN  = '#3fb950'
PURPLE = '#d2a8ff'
ORANGE = '#ffa657'


def _ax_style(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_color(BORDER)
    ax.tick_params(colors='#8b949e', labelsize=8)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TITLE)
    ax.grid(True, color='#21262d', lw=0.5)
    if title:  ax.set_title(title, fontsize=9)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)


# =============================================================================
# Material
# =============================================================================
mat = IM7_8552()
t_ply = 0.125e-3   # [m] cured ply thickness
n_half = 4         # half-stack plies → 8-ply symmetric laminate
h = 2 * n_half * t_ply


# =============================================================================
# SECTION 1 – Stiffness Polars
# =============================================================================
print("=" * 60)
print("SECTION 1 — Stiffness polars")
print("=" * 60)

fig1, axes1 = plt.subplots(1, 2, figsize=(13, 5), facecolor=BG)
fig1.subplots_adjust(wspace=0.35, top=0.88)

stiffness_polar(mat, h_total=h, ax_inplane=axes1[0], ax_bending=axes1[1],
                label='IM7/8552', show=False)

# Annotate optimal angles on in-plane plot
inv   = material_invariants(mat)
thetas = np.linspace(0, 180, 1801)
tr    = np.radians(thetas)
U1, U2, U3, U4, U5 = inv['U1'], inv['U2'], inv['U3'], inv['U4'], inv['U5']

A11 = h * (U1 + U2*np.cos(2*tr) + U3*np.cos(4*tr))
A22 = h * (U1 - U2*np.cos(2*tr) + U3*np.cos(4*tr))
A12 = h * (U4 - U3*np.cos(4*tr))
A66 = h * (U5 - U3*np.cos(4*tr))
A16 = h * (0.5*U2*np.sin(2*tr) + U3*np.sin(4*tr))
A26 = h * (0.5*U2*np.sin(2*tr) - U3*np.sin(4*tr))

Ex_arr = np.array([1.0 / (np.linalg.inv(np.array([
    [A11[i], A12[i], A16[i]],
    [A12[i], A22[i], A26[i]],
    [A16[i], A26[i], A66[i]]]))[0, 0] * h) for i in range(len(thetas))])

theta_peak_Ex = thetas[np.argmax(Ex_arr)]
axes1[0].axvline(theta_peak_Ex, color='#ffffff', lw=0.8, alpha=0.5, ls=':')
axes1[0].text(theta_peak_Ex + 2, axes1[0].get_ylim()[1] * 0.92,
              f'{theta_peak_Ex:.0f}°', color='#8b949e', fontsize=7)

# Mark D16 peaks on bending plot
D16_arr = (h**3/12) * (0.5*U2*np.sin(2*tr) + U3*np.sin(4*tr))
D11_ref = float((h**3/12) * (U1 + U2 + U3))
idx_d16 = np.argmax(np.abs(D16_arr))
axes1[1].axvline(thetas[idx_d16], color='#d2a8ff', lw=0.8, alpha=0.5, ls=':')
axes1[1].text(thetas[idx_d16] + 2, axes1[1].get_ylim()[1] * 0.85,
              f'max |D₁₆|: {thetas[idx_d16]:.0f}°',
              color='#d2a8ff', fontsize=7)

fig1.suptitle('Stiffness Polars — IM7/8552  |  Single-angle laminate [θ]₈',
              color=TITLE, fontsize=11, y=0.98)
out1 = os.path.join(os.path.dirname(__file__), '..', 'outputs',
                    'stiffness_polars.png')
fig1.savefig(out1, dpi=150, bbox_inches='tight', facecolor=BG)
print(f"  Saved → {os.path.abspath(out1)}")


# =============================================================================
# SECTION 2 – Lamination parameter space for common layups
# =============================================================================
print()
print("=" * 60)
print("SECTION 2 — Lamination parameter space")
print("=" * 60)

layups = {
    '[0]₈':            [0,  0,  0,  0,  0,  0,  0,  0],
    '[90]₈':           [90, 90, 90, 90, 90, 90, 90, 90],
    '[±45]₂s':         [45, -45, -45, 45, 45, -45, -45, 45],
    '[0/90]₂s':        [0, 90, 90, 0, 0, 90, 90, 0],
    'QI [±45/0/90]s':  [45, -45, 0, 90, 90, 0, -45, 45],
    '[30/-30/0/90]s':  [30, -30, 0, 90, 90, 0, -30, 30],
    '[15/-15/0/90]s':  [15, -15, 0, 90, 90, 0, -15, 15],
    '[±20/0]s':        [20, -20, 0, 0, 0, 0, -20, 20],
}

print(f"  {'Layup':<22} {'ξ1A':>7} {'ξ2A':>7} {'ξ1D':>7} {'ξ2D':>7} "
      f"{'BTC':>7} {'SEC':>7} {'Sym':>4} {'Bal':>4}")
print("  " + "─" * 78)

lp_data = {}
for name, angles in layups.items():
    plies = [Ply(mat, t_ply, a) for a in angles]
    lam   = Laminate(plies)
    lp    = lamination_parameters(lam)
    btc   = bend_twist_coupling_index(lam)
    sec   = shear_extension_coupling_index(lam)
    lp_data[name] = (lp, btc, sec, lam.is_symmetric, lam.is_balanced)
    print(f"  {name:<22} {lp['xi1A']:>7.3f} {lp['xi2A']:>7.3f} "
          f"{lp['xi1D']:>7.3f} {lp['xi2D']:>7.3f} "
          f"{btc:>7.3f} {sec:>7.3f} "
          f"{'✓' if lam.is_symmetric else '✗':>4} "
          f"{'✓' if lam.is_balanced else '✗':>4}")

# LP feasibility plot + layup scatter
fig2, (ax_lp, ax_btc) = plt.subplots(1, 2, figsize=(11, 5), facecolor=BG)
fig2.subplots_adjust(wspace=0.35, top=0.88)

plot_lp_feasibility(ax=ax_lp, show=False)

# Scatter layup LP points
colours = [BLUE, RED, GREEN, PURPLE, ORANGE, '#79c0ff', '#ffa198', '#56d364']
for (name, (lp, btc, sec, sym, bal)), col in zip(lp_data.items(), colours):
    ax_lp.scatter(lp['xi1A'], lp['xi2A'], s=70, color=col, zorder=6, label=name)
    ax_lp.annotate(name.split('[')[0] or name[:8],
                   (lp['xi1A'], lp['xi2A']),
                   textcoords='offset points', xytext=(5, -4),
                   color=col, fontsize=6.5)
ax_lp.legend(framealpha=0.2, labelcolor=TEXT, fontsize=6.5,
             loc='upper right', ncol=1)

# BTC / SEC bar chart
names_short = [n.replace('_', '\n') for n in layups.keys()]
btc_vals = [lp_data[n][1] for n in layups]
sec_vals = [lp_data[n][2] for n in layups]
x = np.arange(len(layups))
width = 0.38
bars1 = ax_btc.bar(x - width/2, btc_vals, width, label='BTC index', color=PURPLE, alpha=0.8)
bars2 = ax_btc.bar(x + width/2, sec_vals, width, label='SEC index', color=ORANGE, alpha=0.8)
ax_btc.set_xticks(x)
ax_btc.set_xticklabels(list(layups.keys()), rotation=35, ha='right', fontsize=6.5,
                        color=TEXT)
_ax_style(ax_btc, title='Bend-Twist & Shear-Extension Coupling',
          xlabel='Layup', ylabel='Coupling index [-]')
ax_btc.legend(framealpha=0.2, labelcolor=TEXT, fontsize=8)

fig2.suptitle('Lamination Parameter Space — IM7/8552', color=TITLE, fontsize=11)
out2 = os.path.join(os.path.dirname(__file__), '..', 'outputs',
                    'lamination_param_space.png')
fig2.savefig(out2, dpi=150, bbox_inches='tight', facecolor=BG)
print(f"\n  Saved → {os.path.abspath(out2)}")


# =============================================================================
# SECTION 3 – Aeroelastic tailoring: balanced vs unbalanced laminates
# =============================================================================
print()
print("=" * 60)
print("SECTION 3 — Aeroelastic tailoring: balanced vs unbalanced")
print("=" * 60)

# Wing geometry (supersonic transport concept)
wing = WingGeometry(
    semi_span     = 8.0,    # [m]
    root_chord    = 5.0,    # [m]
    taper_ratio   = 0.30,   # tip_chord / root_chord  (1.5/5.0)
    sweep_le_deg  = 35.0,   # [deg] leading-edge sweep
    t_over_c      = 0.05,   # thickness ratio
)

mach       = 1.6
altitude_m = 12_000.0
alpha_deg  = 4.0
n_load     = 2.5

# Construct three laminate candidates:
#   1. Balanced quasi-isotropic [±45/0/90]s   (baseline)
#   2. Moderate unbalanced [30/-45/0/90]s     (D16 ≠ 0, positive BTC)
#   3. Highly unbalanced [15/-60/0/90]s       (larger D16)

lam_configs = {}

# Balanced QI
ang_qi_sym = [45, -45, 0, 90, 90, 0, -45, 45]
plies_qi   = [Ply(mat, t_ply, a) for a in ang_qi_sym]
lam_qi     = Laminate(plies_qi)
lam_configs['Balanced QI\n[±45/0/90]s'] = lam_qi

# Moderate unbalanced
ang_unbal1 = [30, -45, 0, 90, 90, 0, -45, 30]
plies_ub1  = [Ply(mat, t_ply, a) for a in ang_unbal1]
lam_ub1    = Laminate(plies_ub1)
lam_configs['Moderate unbal.\n[30/-45/0/90]s'] = lam_ub1

# More unbalanced
ang_unbal2 = [15, -60, 0, 90, 90, 0, -60, 15]
plies_ub2  = [Ply(mat, t_ply, a) for a in ang_unbal2]
lam_ub2    = Laminate(plies_ub2)
lam_configs['High unbal.\n[15/-60/0/90]s'] = lam_ub2

print(f"\n  {'Config':<28} {'A11 [MN/m]':>11} {'D66 [N·m]':>10} "
      f"{'D16 [N·m]':>10} {'BTC':>7}")
print("  " + "─" * 70)
for name, lam in lam_configs.items():
    btc = bend_twist_coupling_index(lam)
    n   = name.replace('\n', ' ')
    print(f"  {n:<28} {lam.A[0,0]/1e6:>11.2f} {lam.D[2,2]:>10.3f} "
          f"{lam.D[0,2]:>10.3f} {btc:>7.4f}")

# Run aeroelastic analysis for each
ae_results = {}
for name, lam in lam_configs.items():
    D16 = lam.D[0, 2]
    D66 = lam.D[2, 2]
    ae  = static_aeroelastic(
        wing, mach, altitude_m, alpha_deg, n_load,
        laminate_A11=lam.A[0, 0],
        laminate_h  =lam.thickness,
        laminate_D16=D16,
        laminate_D66=D66,
        n_stations  =25,
    )
    ae_results[name] = ae
    relief_pct = ae.aeroelastic_relief * 100
    tip_twist  = float(ae.delta_alpha[-1])
    n_clean    = name.replace('\n', ' ')
    print(f"  {n_clean:<28}  relief = {relief_pct:+.1f}%  "
          f"tip Δα = {tip_twist:+.2f}°  iters = {ae.n_iterations}")

# ── Plots ────────────────────────────────────────────────────────────────────
fig3, axes3 = plt.subplots(2, 2, figsize=(12, 8), facecolor=BG)
fig3.subplots_adjust(hspace=0.38, wspace=0.32, top=0.90)
ax_da, ax_ea, ax_nxx, ax_rel = axes3[0, 0], axes3[0, 1], axes3[1, 0], axes3[1, 1]
for ax in axes3.flat:
    _ax_style(ax)

cols = [BLUE, RED, GREEN]
styles = ['-', '--', '-.']

for (name, ae), col, ls in zip(ae_results.items(), cols, styles):
    eta = ae.etas
    lbl = name.replace('\n', ' ')
    ax_da.plot(eta, ae.delta_alpha, color=col, ls=ls, lw=2, label=lbl)
    ax_ea.plot(eta, ae.alpha_eff,   color=col, ls=ls, lw=2, label=lbl)
    Nyy = np.array([l.Nyy / 1e3 for l in ae.loads_elastic])
    ax_nxx.plot(eta, Nyy, color=col, ls=ls, lw=2, label=lbl)

# Relief bar chart
relief_vals = [ae_results[n].aeroelastic_relief * 100 for n in lam_configs]
bar_labels  = [n.replace('\n', '\n') for n in lam_configs]
bar_cols    = [cols[i] for i in range(len(lam_configs))]
ax_rel.bar(range(len(lam_configs)), relief_vals, color=bar_cols, alpha=0.8)
ax_rel.set_xticks(range(len(lam_configs)))
ax_rel.set_xticklabels(bar_labels, fontsize=7.5, color=TEXT)
ax_rel.axhline(0, color=BORDER, lw=0.8)
_ax_style(ax_rel, title='Tip load relief', ylabel='Aeroelastic relief [%]')

for ax in (ax_da, ax_ea, ax_nxx):
    ax.legend(framealpha=0.2, labelcolor=TEXT, fontsize=7.5)
    ax.set_xlim(0, 1)
    ax.set_xlabel('Span fraction η [-]')

_ax_style(ax_da,  title='Effective AoA change Δα(η)',
          ylabel='Δα [°]')
_ax_style(ax_ea,  title='Effective AoA α_eff(η)',
          ylabel='α_eff [°]')
_ax_style(ax_nxx, title='Chordwise running load N_yy(η)',
          ylabel='$N_{yy}$ [kN/m]')

fig3.suptitle(
    f'Aeroelastic Tailoring — Balanced vs Unbalanced Laminates\n'
    f'M={mach}  |  alt={altitude_m/1000:.0f} km  |  n={n_load}g  |  Λ={wing.sweep_le_deg}°',
    color=TITLE, fontsize=10,
)
out3 = os.path.join(os.path.dirname(__file__), '..', 'outputs',
                    'aeroelastic_tailoring.png')
fig3.savefig(out3, dpi=150, bbox_inches='tight', facecolor=BG)
print(f"\n  Saved → {os.path.abspath(out3)}")


# =============================================================================
# SECTION 4 – LP reconstruction accuracy check
# =============================================================================
print()
print("=" * 60)
print("SECTION 4 — LP reconstruction accuracy")
print("=" * 60)

for name, lam in lam_configs.items():
    lp        = lamination_parameters(lam)
    A_lp, B_lp, D_lp = abd_from_lamination_params(lam.thickness, lp, mat)
    A_err = np.max(np.abs(A_lp - lam.A)) / (np.max(np.abs(lam.A)) + 1e-30)
    D_err = np.max(np.abs(D_lp - lam.D)) / (np.max(np.abs(lam.D)) + 1e-30)
    n = name.replace('\n', ' ')
    print(f"  {n:<28}  max A error = {A_err:.2e}   max D error = {D_err:.2e}")

print()
print("All sections complete.")
print(f"Output images in: {os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'outputs'))}")
