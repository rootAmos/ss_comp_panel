"""
demo_supersonic_panel.py
========================
Composite panel analysis for a supersonic cruise condition.

Mission context: Mach 1.6 cruise at 15,000 m
Panel: upper wing skin at 40% semi-span
Material: IM7/8552 carbon-epoxy (Hexcel)
Stacking: quasi-isotropic [±45/0/90]s

Run:  python demo_supersonic_panel.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

from composite_panel import Ply, Laminate, IM7_8552
from composite_panel import check_laminate
from composite_panel import supersonic_panel_loads

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Mission / panel geometry
# ─────────────────────────────────────────────────────────────────────────────

MACH        = 1.6
ALT_M       = 15_000      # m
ALPHA_DEG   = 3.0
PANEL_CHORD = 0.80        # m   (chordwise dimension of skin panel)
PANEL_SPAN  = 0.50        # m
N_LOAD      = 2.5         # ultimate load factor

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Material & laminate definition
# ─────────────────────────────────────────────────────────────────────────────

PLY_T = 0.125e-3    # 0.125 mm / ply  (standard IM7 prepreg)
mat   = IM7_8552()

# Quasi-isotropic symmetric  [±45/0/90]s  (8 plies total)
angles = [-45, 0, 45, 90, 90, 45, 0, -45]
plies  = [Ply(mat, PLY_T, a) for a in angles]
lam    = Laminate(plies)

# ─────────────────────────────────────────────────────────────────────────────
# 3.  Aerodynamic loads
# ─────────────────────────────────────────────────────────────────────────────

loads = supersonic_panel_loads(
    mach=MACH, altitude_m=ALT_M, alpha_deg=ALPHA_DEG,
    panel_chord=PANEL_CHORD, panel_span=PANEL_SPAN,
    n_load=N_LOAD,
)

print("=" * 65)
print(f"  Supersonic Panel Analysis  –  Mach {MACH} @ {ALT_M/1e3:.0f} km")
print("=" * 65)
print(f"\n  Panel:    {PANEL_CHORD*100:.0f} cm chord × {PANEL_SPAN*100:.0f} cm span")
print(f"  Material: {mat.name}")
print(f"  Layup:    [{'/'.join(str(a) for a in angles)}]")
print(f"  h =       {lam.thickness*1e3:.3f} mm\n")
print(f"  {loads}\n")

# ─────────────────────────────────────────────────────────────────────────────
# 4.  CLT response
# ─────────────────────────────────────────────────────────────────────────────

res = lam.response(N=loads.N, M=loads.M)

print(lam.summary())
print(f"\n  Midplane strains  ε0 = [{', '.join(f'{e*1e6:.2f}' for e in res['eps0'])}]  [με]")
print(f"  Curvatures        κ  = [{', '.join(f'{k:.4f}' for k in res['kappa'])}]  [1/m]")

# ─────────────────────────────────────────────────────────────────────────────
# 5.  Failure analysis
# ─────────────────────────────────────────────────────────────────────────────

results = check_laminate(res, plies, criterion='tsai_wu', verbose=True)

# ─────────────────────────────────────────────────────────────────────────────
# 6.  Stacking sequence trade study  –  governing RF vs angle
# ─────────────────────────────────────────────────────────────────────────────

def _style_ax(ax, bg, fg, grid_color):
    ax.set_facecolor(bg)
    ax.tick_params(colors=fg, labelsize=7)
    ax.xaxis.label.set_color(fg)
    for spine in ax.spines.values():
        spine.set_edgecolor(grid_color)
    ax.grid(axis='x', color=grid_color, linewidth=0.4, alpha=0.5)


print("\n  Running stacking trade study...")
sweep_angles = np.arange(0, 91, 5)
governing_rf = []

for theta in sweep_angles:
    # [θ/-θ/0/90]s
    stack = [theta, -theta, 0, 90, 90, 0, -theta, theta]
    p     = [Ply(mat, PLY_T, a) for a in stack]
    l     = Laminate(p)
    r     = l.response(N=loads.N, M=loads.M)
    fail  = check_laminate(r, p, criterion='tsai_wu', verbose=False)
    gov   = min(fail, key=lambda x: x.rf)
    governing_rf.append(gov.rf)

governing_rf = np.array(governing_rf)

# ─────────────────────────────────────────────────────────────────────────────
# 7.  Plot
# ─────────────────────────────────────────────────────────────────────────────

# Astro Mechanica palette
BG     = '#0a0e1a'
BLUE   = '#00aaff'
WHITE  = '#e8edf5'
GOLD   = '#f0a030'
RED    = '#ff4455'
DIM    = '#3a4060'

fig = plt.figure(figsize=(14, 9), facecolor=BG)
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38,
                         left=0.07, right=0.97, top=0.88, bottom=0.10)

# ── (A) Ply stresses σ1 ──────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
sig1 = [r['ply_stress_12'][k][0] / 1e6 for k in range(len(plies))]
colors = [RED if s < 0 else BLUE for s in sig1]
ax1.barh(range(len(plies)), sig1, color=colors, edgecolor=DIM, linewidth=0.5)
ax1.axvline(0, color=WHITE, linewidth=0.5, alpha=0.4)
ax1.axvline( mat.F1t/1e6, color=BLUE,  linewidth=1, linestyle='--', alpha=0.6, label=f'F1t={mat.F1t/1e6:.0f} MPa')
ax1.axvline(-mat.F1c/1e6, color=RED,   linewidth=1, linestyle='--', alpha=0.6, label=f'F1c={mat.F1c/1e6:.0f} MPa')
ax1.set_yticks(range(len(plies)))
ax1.set_yticklabels([f'{a}°' for a in angles], fontsize=7)
ax1.set_xlabel('σ₁ [MPa]', color=WHITE, fontsize=8)
ax1.set_title('Fibre-dir. stress', color=WHITE, fontsize=9, pad=6)
ax1.legend(fontsize=6, loc='lower right', framealpha=0.2, labelcolor=WHITE)
_style_ax(ax1, BG, WHITE, DIM)

# ── (B) Ply stresses σ2 ──────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
sig2 = [r['ply_stress_12'][k][1] / 1e6 for k in range(len(plies))]
colors2 = [RED if s < 0 else BLUE for s in sig2]
ax2.barh(range(len(plies)), sig2, color=colors2, edgecolor=DIM, linewidth=0.5)
ax2.axvline(0,            color=WHITE, linewidth=0.5, alpha=0.4)
ax2.axvline( mat.F2t/1e6, color=BLUE,  linewidth=1, linestyle='--', alpha=0.6, label=f'F2t={mat.F2t/1e6:.0f} MPa')
ax2.axvline(-mat.F2c/1e6, color=RED,   linewidth=1, linestyle='--', alpha=0.6, label=f'F2c={mat.F2c/1e6:.0f} MPa')
ax2.set_yticks(range(len(plies)))
ax2.set_yticklabels([f'{a}°' for a in angles], fontsize=7)
ax2.set_xlabel('σ₂ [MPa]', color=WHITE, fontsize=8)
ax2.set_title('Transverse stress', color=WHITE, fontsize=9, pad=6)
ax2.legend(fontsize=6, loc='lower right', framealpha=0.2, labelcolor=WHITE)
_style_ax(ax2, BG, WHITE, DIM)

# ── (C) Reserve factors ──────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
rfs = [r.rf for r in results]
bar_colors = [RED if rf < 1.0 else (GOLD if rf < 1.5 else BLUE) for rf in rfs]
ax3.barh(range(len(plies)), rfs, color=bar_colors, edgecolor=DIM, linewidth=0.5)
ax3.axvline(1.0, color=RED,  linewidth=1.5, linestyle='--', alpha=0.8, label='RF = 1.0 (failure)')
ax3.axvline(1.5, color=GOLD, linewidth=1,   linestyle=':',  alpha=0.6, label='RF = 1.5')
ax3.set_yticks(range(len(plies)))
ax3.set_yticklabels([f'{a}°' for a in angles], fontsize=7)
ax3.set_xlabel('Tsai-Wu RF [-]', color=WHITE, fontsize=8)
ax3.set_title('Reserve factors', color=WHITE, fontsize=9, pad=6)
ax3.legend(fontsize=6, framealpha=0.2, labelcolor=WHITE)
_style_ax(ax3, BG, WHITE, DIM)

# ── (D) Stacking trade  ───────────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, :2])
ax4.plot(sweep_angles, governing_rf, color=BLUE, linewidth=2.0)
ax4.fill_between(sweep_angles, 0, governing_rf,
                  where=(governing_rf >= 1.0), alpha=0.12, color=BLUE)
ax4.fill_between(sweep_angles, 0, governing_rf,
                  where=(governing_rf < 1.0),  alpha=0.18, color=RED)
ax4.axhline(1.0, color=RED,  linewidth=1.5, linestyle='--', alpha=0.7, label='RF = 1.0 limit')
ax4.axhline(1.5, color=GOLD, linewidth=1.0, linestyle=':',  alpha=0.5, label='RF = 1.5 target')

# Mark optimum
opt_idx = np.argmax(governing_rf)
ax4.scatter(sweep_angles[opt_idx], governing_rf[opt_idx],
            color=GOLD, s=80, zorder=5,
            label=f'Optimum: θ={sweep_angles[opt_idx]}°, RF={governing_rf[opt_idx]:.2f}')
ax4.set_xlabel('Off-axis ply angle θ [deg]  (layup: [θ/−θ/0/90]s)', color=WHITE, fontsize=9)
ax4.set_ylabel('Governing Tsai-Wu RF [-]', color=WHITE, fontsize=9)
ax4.set_title('Stacking sequence trade  –  governing reserve factor', color=WHITE, fontsize=10, pad=8)
ax4.legend(fontsize=8, framealpha=0.2, labelcolor=WHITE)
ax4.set_xlim(0, 90)
_style_ax(ax4, BG, WHITE, DIM)

# ── (E) Panel load summary ────────────────────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 2])
ax5.axis('off')
load_text = [
    ("Nxx", f"{loads.Nxx/1e3:+.1f} kN/m"),
    ("Nyy", f"{loads.Nyy/1e3:+.1f} kN/m"),
    ("Nxy", f"{loads.Nxy/1e3:+.1f} kN/m"),
    ("Mxx", f"{loads.Mxx:+.2f} N·m/m"),
    ("h",   f"{lam.thickness*1e3:.3f} mm"),
    ("Ex",  f"{lam.Ex/1e9:.1f} GPa"),
    ("Ey",  f"{lam.Ey/1e9:.1f} GPa"),
    ("Min RF", f"{min(rfs):.3f}"),
]
y0 = 0.95
for label, val in load_text:
    color = RED if (label == "Min RF" and min(rfs) < 1.0) else WHITE
    ax5.text(0.05, y0, label, transform=ax5.transAxes,
             color=DIM,   fontsize=9, fontfamily='monospace')
    ax5.text(0.50, y0, val,  transform=ax5.transAxes,
             color=color, fontsize=9, fontfamily='monospace', fontweight='bold')
    y0 -= 0.11
ax5.set_title('Panel summary', color=WHITE, fontsize=9, pad=6)
ax5.set_facecolor(BG)
for spine in ax5.spines.values():
    spine.set_edgecolor(DIM)

# ── Header ────────────────────────────────────────────────────────────────────
fig.text(0.5, 0.955,
         f'Composite Panel Analysis  |  Mach {MACH} @ {ALT_M/1e3:.0f} km  |  '
         f'{mat.name}  [{"/ ".join(str(a) for a in angles)}]',
         ha='center', color=WHITE, fontsize=12, fontweight='bold')
fig.text(0.5, 0.928,
         f'n = {N_LOAD}g  ·  α = {ALPHA_DEG}°  ·  '
         f'{PANEL_CHORD*100:.0f} cm × {PANEL_SPAN*100:.0f} cm panel  ·  '
         f'Tsai-Wu failure criterion',
         ha='center', color=DIM, fontsize=8)

outpath = 'supersonic_panel_analysis.png'
plt.savefig(outpath, dpi=150, bbox_inches='tight', facecolor=BG)
print(f"\n  Figure saved → {outpath}")
plt.close()



