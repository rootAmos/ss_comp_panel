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
