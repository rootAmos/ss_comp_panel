"""
demo_supersonic_panel.py
========================
Classical Laminate Theory (CLT) stress and failure analysis for a composite
upper-wing skin panel at a supersonic cruise condition.

MISSION CONTEXT
---------------
Vehicle   : Supersonic transport / combat aircraft concept
Condition : Mach 1.6 cruise at 15,000 m altitude
Panel     : Upper skin between adjacent rib and stringer (40% semi-span)
Material  : IM7/8552 carbon-epoxy unidirectional prepreg (Hexcel)
Stacking  : Quasi-isotropic  [±45 / 0 / 90]_s   (8 plies, symmetric)

STEPS
-----
1.  Ackeret linearised supersonic theory → aerodynamic running loads
2.  CLT ABD matrix for the chosen laminate
3.  Midplane strains and curvatures (two 3×3 systems, B = 0)
4.  Ply-level stresses in the material (1-2) frame
5.  Tsai-Wu reserve factor for every ply
6.  Sweep off-axis angle θ in [θ/−θ/0/90]_s to find the governing RF

KEY PHYSICS / EQUATIONS (all cited inline below)
-------------------------------------------------
  Ackeret pressure    : ΔCp = 4α / √(M²−1)           [Ackeret 1925]
  CLT membrane        : {N} = [A]{ε₀}                  [Jones 1999, Ch.4]
  CLT bending         : {M} = [D]{κ}                   [Jones 1999, Ch.4]
  Q-bar transform     : Q̄ = T⁻¹ Q (Tᵀ)⁻¹             [Kassapoglou 2013, §2.3]
  Tsai-Wu criterion   : F₁σ₁ + F₂σ₂ + F₁₁σ₁² + F₂₂σ₂² + F₆₆τ₁₂² + 2F₁₂σ₁σ₂ = 1
                                                         [Tsai & Wu 1971]
  RF (smooth form)    : a·RF² + b·RF − 1 = 0  →  RF = (−b + √(b²+4a)) / 2a

References
----------
Ackeret, J.     (1925) Luftkräfte auf Flügel die mit größerer als Schallgeschwindigkeit
                       bewegt werden. ZAMM 5(1), 17–28.
Jones, R.M.     (1999) Mechanics of Composite Materials, 2nd ed. Taylor & Francis.
Kassapoglou, C. (2013) Design and Analysis of Composite Structures. Wiley.
Tsai, S.W. & Wu, E.M. (1971) A General Theory of Strength for Anisotropic Materials.
                       J. Composite Materials 5(1), 58–80.
Anderson, J.D.  (2003) Modern Compressible Flow, 3rd ed. McGraw-Hill.

Run:  python demo_supersonic_panel.py
"""

import sys, os
# Allow importing composite_panel from the local src/ tree
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')   # suppress IPOPT / matplotlib minor warnings

# composite_panel public API
from composite_panel import Ply, Laminate, IM7_8552
from composite_panel import check_laminate          # Tsai-Wu / Tsai-Hill / max-stress
from composite_panel import supersonic_panel_loads  # Ackeret + ISA atmosphere

# =============================================================================
# 1.  MISSION AND PANEL GEOMETRY
# =============================================================================
# These are the fixed inputs that define the design point.
# Changing MACH or ALT_M will recompute all downstream loads automatically.

MACH        = 1.6       # Free-stream Mach number  (supersonic, Ackeret valid ~1.1–4)
ALT_M       = 15_000    # Cruise altitude [m]  — mid-stratosphere, ISA model used
ALPHA_DEG   = 3.0       # Angle of attack [deg]  — typical cruise incidence

# Panel dimensions: distance between adjacent structural members
PANEL_CHORD = 0.80      # Chordwise panel dimension b  [m]  (stringer-to-stringer pitch)
PANEL_SPAN  = 0.50      # Spanwise panel dimension a   [m]  (rib-to-rib pitch)

# Ultimate load factor — FAR 25 transport category limit load × 1.5 safety factor
# Structural sizing always uses ultimate loads (limit × 1.5 = 2.5 for n_limit=1.67g)
N_LOAD      = 2.5       # [-]   n_ultimate = n_limit × 1.5

# =============================================================================
# 2.  MATERIAL AND LAMINATE DEFINITION
# =============================================================================
# IM7/8552 is a high-performance aerospace carbon-epoxy system.
# Nominal ply properties (unidirectional, cured):
#   E1  = 171.4 GPa   (fibre-direction modulus)
#   E2  =   9.08 GPa  (transverse modulus)
#   G12 =   5.29 GPa  (in-plane shear modulus)
#   ν12 =   0.32      (major Poisson's ratio)
#   F1t = 2326 MPa    (longitudinal tensile strength)
#   F1c = 1200 MPa    (longitudinal compressive strength)
#   F2t =   62 MPa    (transverse tensile strength)
#   F2c =  200 MPa    (transverse compressive strength)
#   F6  =  110 MPa    (in-plane shear strength)

PLY_T = 0.125e-3    # Nominal cured ply thickness  [m]  = 0.125 mm  (standard IM7 prepreg)
mat   = IM7_8552()  # Instantiate the IM7/8552 material property object

# ── Quasi-isotropic symmetric layup  [±45 / 0 / 90]_s ──────────────────────
# The "s" denotes a symmetric laminate: the sequence is mirrored about the
# mid-plane.  Writing the half-stack as [-45, 0, 45, 90] and mirroring gives
# the 8-ply full sequence below.
#
# This is a "quasi-isotropic" (QI) laminate because the A-matrix (in-plane
# stiffness) is isotropic — every in-plane direction has the same extensional
# stiffness.  This is a common first-design choice when the load path is not
# yet well established.  A production design would likely deviate from QI to
# add more 0° plies (for bending stiffness) or ±45° plies (for shear).
#
# Reference: Jones (1999) §4.6 — conditions for quasi-isotropy:
#   N ≥ 3 distinct angles, equally spaced by π/N, equal thickness at each angle

angles = [-45, 0, 45, 90,   # bottom half of symmetric stack (z < 0)
           90, 45, 0, -45]  # mirror image (z > 0)  — "s" in [±45/0/90]_s

plies = [Ply(mat, PLY_T, a) for a in angles]   # create 8 Ply objects
lam   = Laminate(plies)                         # assemble ABD matrix

# =============================================================================
# 3.  AERODYNAMIC LOADS  —  ACKERET LINEARISED SUPERSONIC THEORY
# =============================================================================
# Ackeret (1925) derived the pressure coefficient for a flat surface in
# 2-D supersonic flow using linearised small-disturbance theory:
#
#   ΔCp = 4α / √(M²−1)
#
# where α is the angle of attack [rad] and √(M²−1) is the Prandtl-Glauert
# factor for supersonic flow (β_sup).
#
# The net pressure loading on the panel (lower − upper surface) is:
#   Δp = ΔCp × q∞ × n_load,    q∞ = ½ρV²  [Pa]
#
# This Δp then drives the structural running loads:
#   Nyy = −Δp × chord / 2          [N/m]  chordwise compression (upper skin)
#   Nxx =  Nyy × 0.6               [N/m]  spanwise load (beam-flange model)
#   Nxy = |Nxx| × 0.25             [N/m]  torsion → shear flow
#   Mxx =  Δp × chord² / 8         [N·m/m] panel bending (simply-supported)
#
# The ISA atmosphere (built into supersonic_panel_loads) gives ρ and a at ALT_M:
#   Troposphere 0–11 km :  T = 288.15 − 6.5×h/1000  [K],  P = P0·(T/T0)^5.256
#   Stratosphere 11–20 km:  T = 216.65 K (isothermal), P = P11·exp(−g·Δh/RT)
#   Reference: ICAO Doc 7488 — Manual of the ICAO Standard Atmosphere (1993)

loads = supersonic_panel_loads(
    mach        = MACH,
    altitude_m  = ALT_M,
    alpha_deg   = ALPHA_DEG,
    panel_chord = PANEL_CHORD,
    panel_span  = PANEL_SPAN,
    n_load      = N_LOAD,
)

# Print a concise load summary to the console
print("=" * 65)
print(f"  Supersonic Panel Analysis  –  Mach {MACH} @ {ALT_M/1e3:.0f} km")
print("=" * 65)
print(f"\n  Panel:    {PANEL_CHORD*100:.0f} cm chord × {PANEL_SPAN*100:.0f} cm span")
print(f"  Material: {mat.name}")
print(f"  Layup:    [{'/'.join(str(a) for a in angles)}]")
print(f"  h =       {lam.thickness*1e3:.3f} mm\n")
print(f"  {loads}\n")

# =============================================================================
# 4.  CLT RESPONSE  —  MIDPLANE STRAINS AND CURVATURES
# =============================================================================
# Classical Laminate Theory (CLT) solves the constitutive relation:
#
#   ⎡ N ⎤   ⎡ A  B ⎤ ⎡ ε₀ ⎤
#   ⎢   ⎥ = ⎢      ⎥ ⎢    ⎥
#   ⎣ M ⎦   ⎣ B  D ⎦ ⎣  κ ⎦
#
# For a SYMMETRIC laminate, B = 0 identically (coupling matrix vanishes).
# The problem decouples into two independent 3×3 systems:
#   Membrane :  {N} = [A]{ε₀}   →   {ε₀} = [A]⁻¹{N}
#   Bending  :  {M} = [D]{κ}    →   {κ}  = [D]⁻¹{M}
#
# where:
#   A_ij = Σ_k Q̄_ij^(k) · t_k                      (sum over all plies)
#   D_ij = Σ_k Q̄_ij^(k) · (z_k^3 − z_{k-1}^3) / 3  (cubic in ply z-position)
#   Q̄    = transformed reduced stiffness matrix (material frame → global frame)
#
# The Q-bar transformation uses the invariant form (Kassapoglou 2013 §2.4):
#   Q̄₁₁ = U₁ + U₂cos2θ + U₃cos4θ
#   Q̄₂₂ = U₁ − U₂cos2θ + U₃cos4θ
#   Q̄₁₂ = U₄ − U₃cos4θ
#   Q̄₆₆ = U₅ − U₃cos4θ
#   Q̄₁₆ = ½U₂sin2θ + U₃sin4θ
#   Q̄₂₆ = ½U₂sin2θ − U₃sin4θ
# where U₁..U₅ are Tsai-Pagano material invariants (functions of E1,E2,G12,ν12)
# Reference: Jones (1999) §2.14; Kassapoglou (2013) §2.4

res = lam.response(N=loads.N, M=loads.M)   # returns dict: eps0, kappa, ply stresses

# Print midplane strain and curvature state
print(lam.summary())
print(f"\n  Midplane strains  ε₀ = [{', '.join(f'{e*1e6:.2f}' for e in res['eps0'])}]  [με]")
print(f"  Curvatures        κ  = [{', '.join(f'{k:.4f}' for k in res['kappa'])}]  [1/m]")

# =============================================================================
# 5.  FAILURE ANALYSIS  —  TSAI-WU CRITERION
# =============================================================================
# The Tsai-Wu failure criterion (Tsai & Wu 1971) is a tensor polynomial
# interaction criterion for orthotropic materials.  For a plane-stress ply:
#
#   f(σ) = F₁σ₁ + F₂σ₂ + F₁₁σ₁² + F₂₂σ₂² + F₆₆τ₁₂² + 2F₁₂σ₁σ₂ = 1
#
# Failure coefficients (in terms of uniaxial strengths):
#   F₁   = 1/F₁ₜ − 1/F₁c           (linear fibre term)
#   F₂   = 1/F₂ₜ − 1/F₂c           (linear transverse term)
#   F₁₁  = 1/(F₁ₜ · F₁c)           (quadratic fibre term)
#   F₂₂  = 1/(F₂ₜ · F₂c)           (quadratic transverse term)
#   F₆₆  = 1/F₆²                    (quadratic shear term)
#   F₁₂  = −½√(F₁₁ · F₂₂)          (Tsai-Hahn interaction coefficient)
#
# The reserve factor RF is defined as the load scaling factor at first-ply failure:
#   f(RF·σ) = 1   →   F₁₁·RF²·σ₁² + ... + (F₁σ₁ + F₂σ₂)·RF − 1 = 0
#
# Solving the quadratic  a·RF² + b·RF − 1 = 0  (a > 0 always):
#   RF = (−b + √(b² + 4a)) / (2a)      [smooth, branch-free form for AD]
#
# This smooth form avoids if/else on stress sign and is fully differentiable
# through CasADi's automatic differentiation graph.
# Reference: Tsai & Wu (1971); Kassapoglou (2013) §3.5

results = check_laminate(res, plies, criterion='tsai_wu', verbose=True)
# results is a list of PlyFailureResult(ply_index, rf, criterion, stress_12)

# =============================================================================
# 6.  STACKING SEQUENCE TRADE STUDY
# =============================================================================
# Sweeps the off-axis angle θ from 0° to 90° in 5° increments for the
# family  [θ/−θ/0/90]_s  and records the governing (minimum) Tsai-Wu RF.
#
# Physical intuition:
#   θ = 0°  : all fibres are 0° and 90° — good for uniaxial Nxx but poor in shear
#   θ = 45° : maximises in-plane shear stiffness (G_xy peak at ±45°)
#   θ = 90° : same as 0° by symmetry of the family
#
# The ±θ pair is the standard "balance pair" — including +θ and −θ in equal
# proportions makes the A₁₆ and A₂₆ terms zero (no extension-shear coupling).
# This is a standard design rule for production laminates.
# Reference: Jones (1999) §4.5; MIL-HDBK-17-3F §4.3

print("\n  Running stacking trade study...")
sweep_angles = np.arange(0, 91, 5)   # θ = 0, 5, 10, ..., 90 degrees
governing_rf = []

for theta in sweep_angles:
    # Build [θ/−θ/0/90]_s for this angle
    stack = [theta, -theta, 0, 90,   # bottom half
              90,   0, -theta, theta] # symmetric mirror
    p    = [Ply(mat, PLY_T, a) for a in stack]
    l    = Laminate(p)
    r    = l.response(N=loads.N, M=loads.M)
    fail = check_laminate(r, p, criterion='tsai_wu', verbose=False)
    # Governing RF = minimum over all plies (first-ply failure criterion)
    gov  = min(fail, key=lambda x: x.rf)
    governing_rf.append(gov.rf)

governing_rf = np.array(governing_rf)

# =============================================================================
# 7.  FIGURE
# =============================================================================
# Dark aerospace palette.

BG    = '#0a0e1a'   # near-black background
BLUE  = '#00aaff'   # highlight / tensile
WHITE = '#e8edf5'   # text / axes
GOLD  = '#f0a030'   # warning / target RF
RED   = '#ff4455'   # compressive / failure
DIM   = '#3a4060'   # grid / spine colour

# Helper: apply consistent dark-theme styling to an axes object
def _style_ax(ax, bg, fg, grid_color):
    ax.set_facecolor(bg)
    ax.tick_params(colors=fg, labelsize=7)
    ax.xaxis.label.set_color(fg)
    for spine in ax.spines.values():
        spine.set_edgecolor(grid_color)
    ax.grid(axis='x', color=grid_color, linewidth=0.4, alpha=0.5)


fig = plt.figure(figsize=(14, 9), facecolor=BG)
# 2-row, 3-column layout:
#   Row 0: [σ₁ bar chart] [σ₂ bar chart] [RF bar chart]
#   Row 1: [stacking trade study — full width]
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38,
                        left=0.07, right=0.97, top=0.88, bottom=0.10)

# ── Panel A: Ply fibre-direction stress  σ₁  [MPa] ──────────────────────────
# σ₁ is the stress component along the fibre direction in each ply's local
# (1-2) coordinate system.  It is obtained by transforming the global (x-y)
# stress state using the ply angle θ:
#
#   {σ₁₂} = [T(θ)] {σ_xy}
#
# where T(θ) is the standard stress transformation matrix.
# Reference: Jones (1999) Eq. 2.65
#
# Blue bars = tensile (σ₁ > 0), Red bars = compressive (σ₁ < 0).
# Dashed lines show the fibre tensile (F1t) and compressive (F1c) allowables.

ax1 = fig.add_subplot(gs[0, 0])
sig1 = [r['ply_stress_12'][k][0] / 1e6 for k in range(len(plies))]  # Pa → MPa
colors = [RED if s < 0 else BLUE for s in sig1]
ax1.barh(range(len(plies)), sig1, color=colors, edgecolor=DIM, linewidth=0.5)
ax1.axvline(0,              color=WHITE, linewidth=0.5, alpha=0.4)
ax1.axvline( mat.F1t/1e6,  color=BLUE,  linewidth=1, linestyle='--', alpha=0.6,
             label=f'F1t = {mat.F1t/1e6:.0f} MPa  (tensile allowable)')
ax1.axvline(-mat.F1c/1e6,  color=RED,   linewidth=1, linestyle='--', alpha=0.6,
             label=f'F1c = {mat.F1c/1e6:.0f} MPa  (compressive allowable)')
ax1.set_yticks(range(len(plies)))
ax1.set_yticklabels([f'{a}°' for a in angles], fontsize=7)
ax1.set_xlabel('σ₁  [MPa]', color=WHITE, fontsize=8)
ax1.set_title('Fibre-dir. stress  σ₁', color=WHITE, fontsize=9, pad=6)
ax1.legend(fontsize=6, loc='lower right', framealpha=0.2, labelcolor=WHITE)
_style_ax(ax1, BG, WHITE, DIM)

# ── Panel B: Ply transverse stress  σ₂  [MPa] ───────────────────────────────
# σ₂ is the stress perpendicular to the fibres (matrix-dominated).
# The transverse tensile strength F2t = 62 MPa is typically the first
# allowable exceeded in a composite laminate under biaxial loading.
# The large compressive σ₂ values in the 0° plies are a characteristic
# of laminates under biaxial in-plane compression — the constraint from
# adjacent off-axis plies forces the matrix into compression.
# Reference: Jones (1999) §3.3; Kassapoglou (2013) §3.2

ax2 = fig.add_subplot(gs[0, 1])
sig2 = [r['ply_stress_12'][k][1] / 1e6 for k in range(len(plies))]
colors2 = [RED if s < 0 else BLUE for s in sig2]
ax2.barh(range(len(plies)), sig2, color=colors2, edgecolor=DIM, linewidth=0.5)
ax2.axvline(0,              color=WHITE, linewidth=0.5, alpha=0.4)
ax2.axvline( mat.F2t/1e6,  color=BLUE,  linewidth=1, linestyle='--', alpha=0.6,
             label=f'F2t = {mat.F2t/1e6:.0f} MPa  (tensile allowable)')
ax2.axvline(-mat.F2c/1e6,  color=RED,   linewidth=1, linestyle='--', alpha=0.6,
             label=f'F2c = {mat.F2c/1e6:.0f} MPa  (compressive allowable)')
ax2.set_yticks(range(len(plies)))
ax2.set_yticklabels([f'{a}°' for a in angles], fontsize=7)
ax2.set_xlabel('σ₂  [MPa]', color=WHITE, fontsize=8)
ax2.set_title('Transverse stress  σ₂', color=WHITE, fontsize=9, pad=6)
ax2.legend(fontsize=6, loc='lower right', framealpha=0.2, labelcolor=WHITE)
_style_ax(ax2, BG, WHITE, DIM)

# ── Panel C: Tsai-Wu Reserve Factors per ply ─────────────────────────────────
# RF = load scaling factor to reach first-ply failure.
#   RF < 1.0  → ply has already failed under applied load  (RED)
#   RF ∈ [1.0, 1.5)  → structurally intact but below design target  (GOLD)
#   RF ≥ 1.5  → meets the required margin of safety  (BLUE)
#
# Note: the RF of each ply depends on its orientation because the stress state
# in the material (1-2) frame varies with θ.  The governing ply (minimum RF)
# determines whether the laminate as a whole has met its strength requirement.
# Reference: Tsai & Wu (1971); MIL-HDBK-17-3F §4.2.5

ax3 = fig.add_subplot(gs[0, 2])
rfs = [r.rf for r in results]
bar_colors = [RED if rf < 1.0 else (GOLD if rf < 1.5 else BLUE) for rf in rfs]
ax3.barh(range(len(plies)), rfs, color=bar_colors, edgecolor=DIM, linewidth=0.5)
ax3.axvline(1.0, color=RED,  linewidth=1.5, linestyle='--', alpha=0.8,
            label='RF = 1.0  (failure boundary)')
ax3.axvline(1.5, color=GOLD, linewidth=1,   linestyle=':',  alpha=0.6,
            label='RF = 1.5  (design target)')
ax3.set_yticks(range(len(plies)))
ax3.set_yticklabels([f'{a}°' for a in angles], fontsize=7)
ax3.set_xlabel('Tsai-Wu RF  [−]', color=WHITE, fontsize=8)
ax3.set_title('Tsai-Wu reserve factors', color=WHITE, fontsize=9, pad=6)
ax3.legend(fontsize=6, framealpha=0.2, labelcolor=WHITE)
_style_ax(ax3, BG, WHITE, DIM)

# ── Panel D: Stacking trade — governing RF vs off-axis angle θ ──────────────
# This plot answers: "what off-axis angle θ gives the best RF for this load?"
#
# The laminate family is  [θ/−θ/0/90]_s  for θ ∈ [0°, 90°].
#
# Physical interpretation of key angles:
#   θ = 0°  : [0/0/0/90]_s — highly 0°-biased; good fibre-direction RF but
#              the 90° ply carries large transverse tension → governs
#   θ = 45° : [±45/0/90]_s = quasi-isotropic; balanced shear resistance
#   θ → 90° : approaches [90/−90/0/90]_s — very few 0° plies, poor in Nxx
#
# The optimum angle (peak RF) trades off shear capacity from the ±θ plies
# against fibre-direction capacity from the 0° plies.  For a compression-
# dominated load state the optimum typically lies between 30° and 50°.
#
# Blue region  = RF ≥ 1.0 (no failure)
# Red region   = RF < 1.0 (laminate would fail at these load levels)
# Gold dot     = global optimum  arg_max(RF)

ax4 = fig.add_subplot(gs[1, :])    # spans full width
ax4.plot(sweep_angles, governing_rf, color=BLUE, linewidth=2.0)
ax4.fill_between(sweep_angles, 0, governing_rf,
                 where=(governing_rf >= 1.0), alpha=0.12, color=BLUE,
                 label='Structurally viable (RF ≥ 1.0)')
ax4.fill_between(sweep_angles, 0, governing_rf,
                 where=(governing_rf < 1.0),  alpha=0.18, color=RED,
                 label='Failed (RF < 1.0)')
ax4.axhline(1.0, color=RED,  linewidth=1.5, linestyle='--', alpha=0.7,
            label='RF = 1.0  failure boundary')
ax4.axhline(1.5, color=GOLD, linewidth=1.0, linestyle=':',  alpha=0.5,
            label='RF = 1.5  design target')

# Mark the angle that maximises governing RF
opt_idx = np.argmax(governing_rf)
ax4.scatter(sweep_angles[opt_idx], governing_rf[opt_idx],
            color=GOLD, s=80, zorder=5,
            label=f'Optimum: θ = {sweep_angles[opt_idx]}°,  RF = {governing_rf[opt_idx]:.2f}')

ax4.set_xlabel('Off-axis ply angle θ  [deg]    (layup family: [θ/−θ/0/90]_s)',
               color=WHITE, fontsize=9)
ax4.set_ylabel('Governing Tsai-Wu RF  [−]', color=WHITE, fontsize=9)
ax4.set_title('Stacking sequence trade study  —  governing reserve factor vs angle',
              color=WHITE, fontsize=10, pad=8)
ax4.legend(fontsize=8, framealpha=0.2, labelcolor=WHITE)
ax4.set_xlim(0, 90)
_style_ax(ax4, BG, WHITE, DIM)

# ── Figure header ─────────────────────────────────────────────────────────────
fig.text(0.5, 0.955,
         f'Composite Panel Analysis  |  Mach {MACH} @ {ALT_M/1e3:.0f} km  |  '
         f'{mat.name}  [{"/ ".join(str(a) for a in angles)}]',
         ha='center', color=WHITE, fontsize=12, fontweight='bold')
fig.text(0.5, 0.928,
         f'n = {N_LOAD}g  ·  α = {ALPHA_DEG}°  ·  '
         f'{PANEL_CHORD*100:.0f} cm × {PANEL_SPAN*100:.0f} cm panel  ·  '
         f'Tsai-Wu failure criterion  (Tsai & Wu 1971)',
         ha='center', color=DIM, fontsize=8)

# ── Save ──────────────────────────────────────────────────────────────────────
outpath = 'outputs/supersonic_panel_analysis.png'
plt.savefig(outpath, dpi=150, bbox_inches='tight', facecolor=BG)
print(f"\n  Figure saved → {outpath}")
plt.close()
