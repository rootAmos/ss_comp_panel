# `composite_panel.optimizer`

Minimum-mass composite laminate sizing via gradient-based NLP (IPOPT/CasADi).

---

## Problem statement

```
min   ρ · 2 · Σ tₖ                            (areal mass, kg/m²)
 t
s.t.  RF_k(t) ≥ rf_min    ∀ ply k             (Tsai-Wu strength)
      RF_buckle(t) ≥ 1     (if panel dims given) (Whitney buckling)
      tₖ ≥ t_min                               (minimum gauge)
      tᵢ = tⱼ             for ±θ pairs         (balance)
```

Design variables are the half-stack ply thicknesses `t ∈ ℝⁿ`. Symmetry is
enforced by construction (bottom half mirrors top), so B = 0 identically and
the 6×6 ABD system decouples into two 3×3 solves:

```
ε₀ = A⁻¹ N,    κ = D⁻¹ M
```

The objective is linear in `t`. Nonlinearity comes entirely from the constraints:
A is linear in `t`, D is cubic in `t` via the z³ bending integral, and the
Tsai-Wu RF is a rational function of the stresses, which depend on A⁻¹ and D⁻¹.

---

## Implementation notes

### Two numpy namespaces

```python
import aerosandbox.numpy as np   # CasADi-compatible — used inside opti graph
import numpy as _np              # standard numpy — used for constants and post-processing
```

`aerosandbox.numpy` transparently routes operations through CasADi when inputs
are `opti.variable()` expressions. The same `_Q_bar_matrix()` and `_tsai_wu_rf()`
functions therefore work both as plain float calculations and as symbolic
expressions inside the optimizer — no separate CasADi-specific code paths.

### Invariant Q̄ form

`_Q_bar_matrix()` uses the Tsai-Pagano invariant polynomial form rather than
the `T⁻¹ Q Tᵀ` transformation:

```
Q̄₁₁ = Q11·c⁴ + 2(Q12+2Q66)·c²s² + Q22·s⁴
Q̄₁₂ = (Q11+Q22−4Q66)·c²s² + Q12(c⁴+s⁴)
Q̄₆₆ = (Q11+Q22−2Q12−2Q66)·c²s² + Q66(c⁴+s⁴)
Q̄₁₆ = (Q11−Q12−2Q66)·c³s − (Q22−Q12−2Q66)·cs³
... (Kassapoglou 2013, §2.4)
```

This avoids the matrix inversion in `T⁻¹`, which is not clean inside CasADi's
symbolic graph. More importantly, it is differentiable with respect to θ,
enabling angle optimization (`optimize_angles=True`).

### Tsai-Wu RF as a quadratic root

Substituting `σ → RF·σ` into the Tsai-Wu criterion gives:

```
a·RF² + b·RF − 1 = 0

a = F₁₁σ₁² + F₂₂σ₂² + F₆₆τ₁₂² + 2F₁₂σ₁σ₂
b = F₁σ₁ + F₂σ₂

RF = (−b + √(b² + 4a)) / (2a + ε)
```

The `ε` regularisation prevents 0/0 at zero-stress states and keeps the
expression smooth everywhere — required for CasADi's AD to generate valid
Jacobians. Implemented in `_tsai_wu_rf()`.

### Buckling

Uses `buckling_rf_smooth()` from `composite_panel.buckling` — the Seydel/Whitney
approximation with fixed mode numbers (no `min` over integers, which would be
non-differentiable). D enters cubically in the z³ bending integral, so the
buckling constraint is nonlinear in `t` even though the load is fixed.

### Thermal loads

If a `ThermalState` is provided, `N_T` and `M_T` are computed once at the
initial-guess geometry and added to the mechanical loads before the optimizer
runs. This approximation is adequate for preliminary sizing — the coupling
between ply thickness and `z_mid` in the thermal resultant integral is
second-order compared to the ΔT contribution.

---

## Multi-case NLP

`optimize_laminate_multicase()` adds one Tsai-Wu constraint block per load case:

```
n_plies × n_cases  constraints:  RF_k,c(t) ≥ rf_min   ∀ k, c
```

All share the same `t` variables, so IPOPT returns the minimum-mass laminate
that satisfies the full load envelope simultaneously. Running
`optimize_laminate()` per case and taking the envelope is non-conservative —
the governing ply typically differs between cases.

The post-processing step identifies which case has the lowest RF per ply
(`governing_cases`) and the minimum RF per case across all plies (`rf_per_case`).

---

## Wing-level sizing

`optimize_wing()` calls `optimize_laminate()` independently at `n_stations`
spanwise stations η ∈ [0.05, 0.95]. Panels are not structurally coupled —
continuity of ply thicknesses between stations is not enforced. Skin mass
is integrated via:

```
m = ∫₀ᵇ ρ·h(η)·c(η) dη  ≈  trapezoid(ρ·h[i]·c[i],  η·b)
```

---

## Data flow

```
Flight condition
      │
      ▼
wing_panel_loads() / LoadsDatabase.filter_eta()
      │  N, M per station / case
      ▼
optimize_laminate() or optimize_laminate_multicase()
      │
      ├─ _Q_bar_matrix(θ)          Q̄ₖ  [symbolic if optimize_angles]
      ├─ _build_ABD_symmetric(t)   A(t), D(t)  [symbolic, D cubic in t]
      ├─ A⁻¹N, D⁻¹M               ε₀, κ  [symbolic]
      ├─ _tsai_wu_rf(σ₁₂)         RF_k(t)  [smooth, differentiable]
      ├─ buckling_rf_smooth(D)     RF_buckle(t)
      └─ opti.solve()              IPOPT → t*  (10–30 Newton iterations)
              │
              ▼
      OptimizationResult / MulticaseOptimizationResult / WingOptimizationResult
```

---

## Aeroelastic tailoring

`optimize_laminate_aeroelastic()` extends the base NLP with a **CasADi-native
aeroelastic washout constraint** — no inner loop, no finite-difference
perturbation:

```
min   ρ · 2 · Σ tₖ
 t,θ
s.t.  RF_k(t,θ) ≥ rf_min         ∀ ply k    (Tsai-Wu strength)
      RF_buckle(t) ≥ 1            (if panel dims given)
      Δα_tip [deg] ≤ −relief_min             (aeroelastic washout)
      tₖ ≥ t_min
      θₖ ∈ [θ_lo, θ_hi]          (if use_bt_coupling=True)
```

When `use_bt_coupling=True`, ply angles become additional design variables and
balance constraints are lifted, allowing D16 ≠ 0.

### Aeroelastic constraint derivation

Wing-box stiffnesses are expressed directly from the CasADi A/D matrices:

```
E_eff = A₁₁ / h_total            [Pa]
EI    = 2 · E_eff · b_box · (h_box/2)²   [N·m²]   (bending)
GJ    = 4 · D₆₆ · b_box                  [N·m²]   (torsional)
EK    = 2 · D₁₆ · b_box                  [N·m²]   (bend-twist coupling)
```

Tip slope (elliptic lift distribution, exact closed-form):

```
θ_tip = W_semi · L² / (8 · EI)            [rad]
```

The factor 1/8 comes from integrating q(y) = q₀√(1−(y/L)²) through the
Euler-Bernoulli cantilever:

```
θ_tip = (1/EI) ∫₀ᴸ ∫_y^L q(s)(s−y) ds dy  =  q₀ · L³ · π / (32 · EI)
```

with q₀ = 4W/(πL).

Total tip washout (sweep geometry + bend-twist coupling):

```
Δα_tip = −θ_tip · (tan Λ + EK/GJ)         [rad]
```

All three stiffnesses (EI, GJ, EK) are differentiable with respect to ply
thicknesses `t` (through A₁₁ and D₆₆) and ply angles `θ` (through D₁₆ and
the Q̄-invariant form). IPOPT therefore receives exact Jacobians for the
aeroelastic constraint from CasADi's AD — the same as for Tsai-Wu and buckling.

### Physical interpretation

| Laminate type | D₁₆ | Washout mechanism |
|---|---|---|
| Balanced `[0/±45/90]s` | 0 | Geometric only — sweep + EI compliance |
| Unbalanced (free angles) | ≠ 0 | Sweep + EK/GJ augmentation |

For a swept-back wing, EK > 0 (positive D₁₆) **augments** the geometric
washout. The optimizer can therefore achieve the required Δα_tip at **higher**
EI (thicker, stronger panels) than the geometry-only case, reducing the mass
penalty from the aeroelastic constraint.

The `demo_aeroelastic_tailoring.py` script shows the three-case comparison:
strength-only baseline (+0%), geometry washout (+33%), bend-twist coupled (+18%).

### Data flow

```
Flight condition + WingGeometry
         │
         ▼
optimize_laminate_aeroelastic(N, M, mat, angles, wing, n_load, relief_min)
         │
         ├─ Tsai-Wu constraints          (same as optimize_laminate)
         ├─ Buckling constraint          (same as optimize_laminate)
         │
         ├─ A₁₁, D₁₆, D₆₆  ← _build_ABD_symmetric(t, Q_bar(θ))  [CasADi]
         ├─ EI, GJ, EK      ← stiffness expressions               [CasADi]
         ├─ θ_tip           ← W_semi·L²/(8·EI)                    [CasADi]
         └─ Δα_tip ≤ −relief_min                                   [CasADi]
                  │
                  └─ opti.solve()  →  AeroelasticOptimizationResult
```

---

## Sensitivity analysis

The parametric sensitivity of optimal mass to any flight parameter is available
analytically from the solved KKT conditions — one solve yields N sensitivities:

```
d(m*)/d(n_load)  — load factor sensitivity
d(m*)/d(Mach)    — Mach sensitivity
```

`demo_sensitivity.py` builds explicit sweep curves across n_load ∈ [1.5, 4.0]g
and Mach ∈ [1.4, 4.0] and annotates the slope at the nominal design point.
Normalised log-log sensitivities at the nominal point (n=2.5g, M=2.4, Alt=25.9km):

| Parameter | ∂(ln m*)/∂(ln p) | Interpretation |
|---|---|---|
| n_load | 0.40 | Doubling load factor → +33% mass |
| Mach | 0.16 | Doubling Mach → +12% mass |

Load factor dominates; Mach matters primarily above M3.5 where dynamic
pressure growth accelerates.

---

## References

| | |
|---|---|
| CLT, ABD | Jones — *Mechanics of Composite Materials* (1999) Ch. 4 |
| Q̄ invariant form | Kassapoglou — *Design and Analysis of Composite Structures* (2013) §2.4 |
| Tsai-Wu | Tsai & Wu — *J. Composite Materials* 5(1), 1971 |
| Buckling (compression) | Timoshenko & Gere — *Theory of Elastic Stability* (1961) Ch. 9 |
| Buckling (shear, Seydel) | ESDU 02.03.11 |
| Buckling (interaction) | Whitney — *Structural Analysis of Laminated Anisotropic Plates* (1987) |
| Ackeret pressure | Ackeret — *ZAMM* 5(1), 1925 |
