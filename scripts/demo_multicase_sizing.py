"""
demo_multicase_sizing.py
========================
Flight-envelope laminate sizing: single-condition vs multi-case NLP.

THE CORE PROBLEM
----------------
Single-case sizing is not conservative for combined loading envelopes.
A laminate sized only for the worst Nxx case under-sizes plies that carry
Nyy.  A laminate sized only for the worst Nyy case under-sizes plies that
carry Nxx.  The two worst conditions may come from completely different
flight regimes.

Three sizing approaches compared:

  1. Single worst-case Nxx  →  lightest, but fails under pressure-dominated cases
  2. Single worst-case Nyy  →  lightest, but fails under bending-dominated cases
  3. Multi-case NLP         →  all cases as simultaneous Tsai-Wu constraints;
                               one optimizer, correct result

PART A — Conceptual illustration
  Synthetic 'bending-dominated' vs 'pressure-dominated' load cases show
  why the multi-case laminate is heavier than either single-case result.

PART B — Flight envelope application
  Real load cases from scripts/flight_envelope.csv (generated from the oblique-shock
  / Prandtl-Glauert aero model across the Mach envelope).  Shows governing
  case per ply at each span station.

Run:
    python -X utf8 demo_multicase_sizing.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as _np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

from composite_panel import IM7_8552, LoadCase, LoadsDatabase, Ply, Laminate
from composite_panel import check_laminate
from composite_panel.optimizer import (
    optimize_laminate, optimize_laminate_multicase,
    detect_balance_pairs,
)

# ── Style ─────────────────────────────────────────────────────────────────────
BG    = "#0a0e1a"; WHITE = "#e8edf5"; DIM = "#3a4060"
C_SGL_A = "#4488ff"    # single: bending
C_SGL_B = "#ff8833"    # single: pressure
C_MULTI = "#00ddbb"    # multi-case
C_FAIL  = "#ff4455"    # failed RF check
C_PASS  = "#44cc88"    # passed RF check

def _style(ax, legend=False):
    ax.set_facecolor(BG)
    ax.tick_params(colors=WHITE, labelsize=8)
    ax.xaxis.label.set_color(WHITE); ax.yaxis.label.set_color(WHITE)
    ax.title.set_color(WHITE)
    for sp in ax.spines.values(): sp.set_edgecolor(DIM)
    ax.grid(color=DIM, linewidth=0.4, alpha=0.5)
    if legend:
        ax.legend(fontsize=7, framealpha=0.15, labelcolor=WHITE)

mat    = IM7_8552()
angles = [0.0, 45.0, -45.0, 90.0]
pairs  = detect_balance_pairs(angles)
RF_MIN = 1.5

ang_full  = angles + list(reversed(angles))
n_full    = len(ang_full)
ply_lbls  = [f"ply{k}\n{ang_full[k]:+.0f}°" for k in range(n_full)]

# ═══════════════════════════════════════════════════════════════════════════════
# PART A — Conceptual illustration: bending vs pressure conflict
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("  PART A — Conceptual: bending vs pressure conflict")
print("=" * 65)

# Case A: bending-dominated (high-g subsonic at low altitude)
#   Large Nxx (spanwise compression from wing bending moment)
#   Moderate Nyy (dynamic pressure × ΔCp is moderate at M0.8)
#   → optimizer grows 0° plies to carry Nxx; 90° plies stay at t_min
case_bend = LoadCase(
    name        = "bending_dom",
    Nxx         = -400e3,   # N/m  — large spanwise compression
    Nyy         = -15e3,    # N/m  — small chordwise compression
    Nxy         =  10e3,    # N/m
    source      = "synthetic",
    description = "High-g subsonic: bending-dominated",
)

# Case B: pressure-dominated (high-speed supersonic at 1g)
#   Small Nxx (low load factor → small bending moment)
#   Large Nyy (very high dynamic pressure at M2–3 dominates chordwise)
#   → optimizer grows 90° plies to carry Nyy; 0° plies stay at t_min
case_pres = LoadCase(
    name        = "pressure_dom",
    Nxx         = -50e3,    # N/m  — small spanwise compression
    Nyy         = -300e3,   # N/m  — large chordwise compression
    Nxy         =  15e3,    # N/m
    source      = "synthetic",
    description = "High-speed 1g: pressure-dominated",
)

# ── Single-case results ────────────────────────────────────────────────────────
r_bend  = optimize_laminate(case_bend.N, case_bend.M, mat, angles,
                            balance_pairs=pairs, rf_min=RF_MIN, verbose=False)
r_pres  = optimize_laminate(case_pres.N, case_pres.M, mat, angles,
                            balance_pairs=pairs, rf_min=RF_MIN, verbose=False)
r_multi_A = optimize_laminate_multicase([case_bend, case_pres], mat, angles,
                                        balance_pairs=pairs, rf_min=RF_MIN,
                                        verbose=False)

print(f"\n  Case A — bending-dominated:   {case_bend}")
print(f"  Case B — pressure-dominated:  {case_pres}")
print(f"\n  Sizing results:")
print(f"  {'Method':<32}  {'h [mm]':>8}  {'rho*h [kg/m²]':>14}  t_half")
print(f"  {'─'*32}  {'─'*8}  {'─'*14}  ─────")
for label, r in [
        ("Single: Case A (bending)", r_bend),
        ("Single: Case B (pressure)", r_pres),
        ("Multi-case NLP (A + B)", r_multi_A.base),
]:
    t_str = "  ".join(f"{t*1e6:.0f}µm" for t in r.t_half)
    print(f"  {label:<32}  {r.total_h*1e3:>8.2f}  {r.areal_density:>14.3f}  [{t_str}]")

delta_vs_bend = (r_multi_A.total_h - r_bend.total_h) / r_bend.total_h * 100
delta_vs_pres = (r_multi_A.total_h - r_pres.total_h) / r_pres.total_h * 100
print(f"\n  Multi-case is {delta_vs_bend:+.1f}% vs single-A,  "
      f"{delta_vs_pres:+.1f}% vs single-B")
print(f"  Governing case per ply:")
for k, (ang, gov) in enumerate(zip(ang_full, r_multi_A.governing_cases)):
    print(f"    ply {k:2d}  θ={ang:+5.0f}°  →  {gov}")

# ── Cross-check: do single-case laminates survive the other case? ─────────────
print(f"\n  RF cross-check: does each single-case laminate survive the other?")
for lam_label, r_lam in [("Bending-sized laminate", r_bend),
                          ("Pressure-sized laminate", r_pres)]:
    lam = Laminate([Ply(mat, r_lam.t_full[k], ang_full[k]) for k in range(n_full)])
    for case_label, case in [("Case A (bending)", case_bend),
                              ("Case B (pressure)", case_pres)]:
        resp  = lam.response(N=case.N, M=case.M)
        fails = check_laminate(resp, lam.plies, criterion="tsai_wu", verbose=False)
        min_rf = min(f.rf for f in fails)
        flag = "OK  " if min_rf >= RF_MIN else "FAIL"
        print(f"    [{flag}]  {lam_label:<28}  under {case_label}  →  RF = {min_rf:.3f}")


# ═══════════════════════════════════════════════════════════════════════════════
# PART B — Flight envelope CSV
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("  PART B — Flight envelope from CSV")
print(f"{'='*65}")

db_full = LoadsDatabase.from_csv("scripts/flight_envelope.csv")
db_full.print_summary()

ETA_MAP   = {"root": 0.15, "mid": 0.45, "tip": 0.75}
ETA_LABELS = {"root": "root  η=0.15", "mid": "mid   η=0.45", "tip": "tip   η=0.75"}

results_B = {}   # {station: MulticaseOptimizationResult}

for key, eta in ETA_MAP.items():
    db = db_full.filter_eta(eta, tol=0.01)
    worst_case = min(db.cases, key=lambda c: c.Nxx)
    print(f"\n  ── {ETA_LABELS[key]}  ({len(db)} cases) ──")

    r_single = optimize_laminate(
        worst_case.N, worst_case.M, mat, angles,
        balance_pairs=pairs, rf_min=RF_MIN, verbose=False,
    )
    r_multi = optimize_laminate_multicase(
        db, mat, angles, balance_pairs=pairs, rf_min=RF_MIN, verbose=False,
    )
    results_B[key] = {"single": r_single, "multi": r_multi,
                      "db": db, "worst": worst_case}

    delta = (r_multi.total_h - r_single.total_h) / r_single.total_h * 100
    print(f"  Single ({worst_case.name}):  h={r_single.total_h*1e3:.2f}mm  "
          f"rho*h={r_single.areal_density:.3f} kg/m²")
    print(f"  Multi-case ({len(db)} cases):  h={r_multi.total_h*1e3:.2f}mm  "
          f"rho*h={r_multi.areal_density:.3f} kg/m²  ({delta:+.1f}%)")
    print(f"  Governing cases:  " + ", ".join(
        f"ply{k}({ang_full[k]:+.0f}°)→{gov}"
        for k, gov in enumerate(r_multi.governing_cases)
        if k < 4  # show half-stack only
    ))


# ═══════════════════════════════════════════════════════════════════════════════
# Figure
# ═══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 10), facecolor=BG)
gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.52, wspace=0.38,
                        left=0.06, right=0.98, top=0.88, bottom=0.09)

# ── (A1) Ply thicknesses: single vs multi for conceptual case ─────────────────
ax_A1 = fig.add_subplot(gs[0, 0])
x    = _np.arange(n_full)
w    = 0.28
ax_A1.bar(x - w, r_bend.t_full * 1e6,  width=w, color=C_SGL_A, label="Single A (bend)", alpha=0.85)
ax_A1.bar(x,     r_pres.t_full * 1e6,  width=w, color=C_SGL_B, label="Single B (pres)", alpha=0.85)
ax_A1.bar(x + w, r_multi_A.t_full * 1e6, width=w, color=C_MULTI, label="Multi-case NLP", alpha=0.85)
ax_A1.set_xticks(x); ax_A1.set_xticklabels(ply_lbls, fontsize=6)
ax_A1.set_ylabel("Ply thickness  [µm]", fontsize=8)
ax_A1.set_title("Ply thickness: conceptual\nbending vs pressure conflict", fontsize=9, pad=5)
_style(ax_A1, legend=True)

# ── (A2) RF cross-check grid for conceptual case ──────────────────────────────
ax_A2 = fig.add_subplot(gs[0, 1:3])
cross_labels_x = ["Case A\n(bend)", "Case B\n(pres)"]
cross_labels_y = ["Single A\n(bend-sized)", "Single B\n(pres-sized)", "Multi-case\nNLP"]
rfs = _np.zeros((3, 2))
for li, (lab, r_lam) in enumerate([("Single A", r_bend),
                                     ("Single B", r_pres),
                                     ("Multi",    r_multi_A.base)]):
    lam = Laminate([Ply(mat, r_lam.t_full[k], ang_full[k]) for k in range(n_full)])
    for ci, case in enumerate([case_bend, case_pres]):
        resp  = lam.response(N=case.N, M=case.M)
        fails = check_laminate(resp, lam.plies, criterion="tsai_wu", verbose=False)
        rfs[li, ci] = min(f.rf for f in fails)

im = ax_A2.imshow(rfs, vmin=0.9, vmax=2.5,
                  cmap="RdYlGn", aspect="auto")
for i in range(3):
    for j in range(2):
        col = "black" if rfs[i, j] > 1.5 else "white"
        flag = "✓" if rfs[i, j] >= RF_MIN else "✗"
        ax_A2.text(j, i, f"{flag}\n{rfs[i,j]:.2f}", ha="center", va="center",
                   fontsize=9, color=col, fontweight="bold")
ax_A2.set_xticks([0, 1]); ax_A2.set_xticklabels(cross_labels_x, fontsize=8, color=WHITE)
ax_A2.set_yticks([0, 1, 2]); ax_A2.set_yticklabels(cross_labels_y, fontsize=8, color=WHITE)
ax_A2.set_title("RF cross-check\n(red = fails constraint)", fontsize=9, pad=5)
ax_A2.title.set_color(WHITE)
ax_A2.set_facecolor(BG)

# ── (A3) Mass comparison bar: conceptual ──────────────────────────────────────
ax_A3 = fig.add_subplot(gs[0, 3])
methods_A = ["Single A\n(bending)", "Single B\n(pressure)", "Multi-case\nNLP"]
masses_A  = [r_bend.areal_density, r_pres.areal_density, r_multi_A.areal_density]
colors_A  = [C_SGL_A, C_SGL_B, C_MULTI]
bars = ax_A3.bar(methods_A, masses_A, color=colors_A, edgecolor=DIM, alpha=0.85)
ax_A3.bar_label(bars, fmt="%.3f kg/m²", color=WHITE, fontsize=7, padding=3)
ax_A3.axhline(r_multi_A.areal_density, color=C_MULTI, lw=1.2, linestyle="--", alpha=0.6)
ax_A3.set_ylabel("Areal density  [kg/m²]", fontsize=8)
ax_A3.set_title("Mass comparison\nconceptual case", fontsize=9, pad=5)
_style(ax_A3)

# ── (B1-B3) Flight envelope — mass comparison per station ─────────────────────
for col, key in enumerate(["root", "mid", "tip"]):
    ax = fig.add_subplot(gs[1, col])
    r  = results_B[key]
    hs = [r["single"].areal_density, r["multi"].areal_density]
    lbs = [f"Single\n({r['worst'].name})", f"Multi-case\n({len(r['db'])} cases)"]
    clrs = [C_SGL_A, C_MULTI]
    bars = ax.bar(lbs, hs, color=clrs, edgecolor=DIM, alpha=0.85)
    ax.bar_label(bars, fmt="%.3f", color=WHITE, fontsize=8, padding=3)
    ax.set_ylabel("Areal density  [kg/m²]", fontsize=8)
    ax.set_title(f"Flight envelope — {ETA_LABELS[key]}", fontsize=9, pad=5)
    # Annotate governing cases on the multi bar
    gov_unique = list(dict.fromkeys(r["multi"].governing_cases))
    ann = " | ".join(gov_unique[:3])
    if len(gov_unique) > 3:
        ann += f" +{len(gov_unique)-3}"
    ax.text(1, hs[1] * 0.5, ann, ha="center", va="center",
            fontsize=6, color=WHITE, alpha=0.8, wrap=True)
    _style(ax)

# ── (B4) RF per case at mid station ───────────────────────────────────────────
ax_B4 = fig.add_subplot(gs[1, 3])
rfpc = results_B["mid"]["multi"].rf_per_case
names_s = sorted(rfpc, key=lambda n: rfpc[n])
rf_s    = [rfpc[n] for n in names_s]
bc4     = [C_FAIL if r < RF_MIN * 1.1 else C_MULTI for r in rf_s]
bars_b4 = ax_B4.barh(names_s, rf_s, color=bc4, edgecolor=DIM, alpha=0.85)
ax_B4.axvline(RF_MIN, color=WHITE, lw=1.2, linestyle="--", alpha=0.7,
              label=f"RF_min={RF_MIN}")
ax_B4.bar_label(bars_b4, fmt="%.3f", color=WHITE, fontsize=6, padding=2)
ax_B4.set_xlabel("Min Tsai-Wu RF  (across all plies)", fontsize=8)
ax_B4.set_title("RF per case — mid-span\nflight envelope (multi-case NLP)", fontsize=9, pad=5)
_style(ax_B4, legend=True)

# ── Header ─────────────────────────────────────────────────────────────────────
fig.text(0.5, 0.955,
         "Flight Envelope Sizing  |  IM7/8552  [0/+45/-45/90]s  |  RF_min ≥ 1.5",
         ha="center", color=WHITE, fontsize=12, fontweight="bold")
fig.text(0.5, 0.928,
         "Left: why single-case sizing is non-conservative (conceptual)  "
         "│  Right: flight envelope from CSV",
         ha="center", color=DIM, fontsize=8)

outpath = "outputs/multicase_sizing.png"
plt.savefig(outpath, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"\n  Figure saved → {outpath}")
plt.close()
