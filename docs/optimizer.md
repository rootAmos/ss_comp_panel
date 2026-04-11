# `composite_panel.optimizer`

Minimum-mass composite laminate sizing via gradient-based NLP (IPOPT/CasADi).

---

## Problem statement

```
min   rho * 2 * Sigma t_k                            (areal mass, kg/m^2)
 t
s.t.  RF_k(t) >= rf_min    for all ply k             (Tsai-Wu strength)
      RF_buckle(t) >= 1     (if panel dims given) (Whitney buckling)
      t_k >= t_min                               (minimum gauge)
      t_i = t_j             for +/-theta pairs         (balance)
```

Design variables are the half-stack ply thicknesses `t in R^n`. Symmetry is
enforced by construction (bottom half mirrors top), so B = 0 identically and
the 6x6 ABD system decouples into two 3x3 solves:

```
eps_0 = A^-1 N,    kappa = D^-1 M
```

The objective is linear in `t`. Nonlinearity comes entirely from the constraints:
A is linear in `t`, D is cubic in `t` via the z^3 bending integral, and the
Tsai-Wu RF is a rational function of the stresses, which depend on A^-1 and D^-1.

---

## Implementation notes

### Two numpy namespaces

```python
import aerosandbox.numpy as np   # CasADi-compatible  --  used inside opti graph
import numpy as _np              # standard numpy  --  used for constants and post-processing
```

`aerosandbox.numpy` transparently routes operations through CasADi when inputs
are `opti.variable()` expressions. The same `_Q_bar_matrix()` and `_tsai_wu_rf()`
functions therefore work both as plain float calculations and as symbolic
expressions inside the optimizer  --  no separate CasADi-specific code paths.

### Invariant Q form

`_Q_bar_matrix()` uses the Tsai-Pagano invariant polynomial form rather than
the `T^-1 Q TT` transformation:

```
Q_1_1 = Q11*c^4 + 2(Q12+2Q66)*c^2s^2 + Q22*s^4
Q_1_2 = (Q11+Q22-4Q66)*c^2s^2 + Q12(c^4+s^4)
Q_6_6 = (Q11+Q22-2Q12-2Q66)*c^2s^2 + Q66(c^4+s^4)
Q_1_6 = (Q11-Q12-2Q66)*c^3s - (Q22-Q12-2Q66)*cs^3
... (Kassapoglou 2013, Sec.2.4)
```

This avoids the matrix inversion in `T^-1`, which is not clean inside CasADi's
symbolic graph. More importantly, it is differentiable with respect to theta,
enabling angle optimization (`optimize_angles=True`).

### Tsai-Wu RF as a quadratic root

Substituting `sigma -> RF*sigma` into the Tsai-Wu criterion gives:

```
a*RF^2 + b*RF - 1 = 0

a = F_1_1sigma_1^2 + F_2_2sigma_2^2 + F_6_6tau_1_2^2 + 2F_1_2sigma_1sigma_2
b = F_1sigma_1 + F_2sigma_2

RF = (-b + sqrt(b^2 + 4a)) / (2a + eps)
```

The `eps` regularisation prevents 0/0 at zero-stress states and keeps the
expression smooth everywhere  --  required for CasADi's AD to generate valid
Jacobians. Implemented in `_tsai_wu_rf()`.

### Buckling

Uses `buckling_rf_smooth()` from `composite_panel.buckling`  --  the Seydel/Whitney
approximation with fixed mode numbers (no `min` over integers, which would be
non-differentiable). D enters cubically in the z^3 bending integral, so the
buckling constraint is nonlinear in `t` even though the load is fixed.

### Thermal loads

If a `ThermalState` is provided, `N_T` and `M_T` are computed once at the
initial-guess geometry and added to the mechanical loads before the optimizer
runs. This approximation is adequate for preliminary sizing  --  the coupling
between ply thickness and `z_mid` in the thermal resultant integral is
second-order compared to the DeltaT contribution.

---

## Multi-case NLP

`optimize_laminate_multicase()` adds one Tsai-Wu constraint block per load case:

```
n_plies x n_cases  constraints:  RF_k,c(t) >= rf_min   for all k, c
```

All share the same `t` variables, so IPOPT returns the minimum-mass laminate
that satisfies the full load envelope simultaneously. Running
`optimize_laminate()` per case and taking the envelope is non-conservative  -- 
the governing ply typically differs between cases.

The post-processing step identifies which case has the lowest RF per ply
(`governing_cases`) and the minimum RF per case across all plies (`rf_per_case`).

---

## Wing-level sizing

`optimize_wing()` calls `optimize_laminate()` independently at `n_stations`
spanwise stations eta in [0.05, 0.95]. Panels are not structurally coupled  -- 
continuity of ply thicknesses between stations is not enforced. Skin mass
is integrated via:

```
m = integral_0b rho*h(eta)*c(eta) deta  ~=  trapezoid(rho*h[i]*c[i],  eta*b)
```

---

## Data flow

```
Flight condition
      |
      v
wing_panel_loads() / LoadsDatabase.filter_eta()
      |  N, M per station / case
      v
optimize_laminate() or optimize_laminate_multicase()
      |
      |--- _Q_bar_matrix(theta)          Q_k  [symbolic if optimize_angles]
      |--- _build_ABD_symmetric(t)   A(t), D(t)  [symbolic, D cubic in t]
      |--- A^-1N, D^-1M               eps_0, kappa  [symbolic]
      |--- _tsai_wu_rf(sigma_1_2)         RF_k(t)  [smooth, differentiable]
      |--- buckling_rf_smooth(D)     RF_buckle(t)
      +--- opti.solve()              IPOPT -> t*  (10-30 Newton iterations)
              |
              v
      OptimizationResult / MulticaseOptimizationResult / WingOptimizationResult
```

---

## Aeroelastic tailoring

`optimize_laminate_aeroelastic()` extends the base NLP with a **CasADi-native
aeroelastic washout constraint**  --  no inner loop, no finite-difference
perturbation:

```
min   rho * 2 * Sigma t_k
 t,theta
s.t.  RF_k(t,theta) >= rf_min         for all ply k    (Tsai-Wu strength)
      RF_buckle(t) >= 1            (if panel dims given)
      Deltaalpha_tip [deg] <= -relief_min             (aeroelastic washout)
      t_k >= t_min
      theta_k in [theta_lo, theta_hi]          (if use_bt_coupling=True)
```

When `use_bt_coupling=True`, ply angles become additional design variables and
balance constraints are lifted, allowing D16 != 0.

### Aeroelastic constraint derivation

Wing-box stiffnesses are expressed directly from the CasADi A/D matrices:

```
E_eff = A_1_1 / h_total            [Pa]
EI    = 2 * E_eff * b_box * (h_box/2)^2   [N*m^2]   (bending)
GJ    = 4 * D_6_6 * b_box                  [N*m^2]   (torsional)
EK    = 2 * D_1_6 * b_box                  [N*m^2]   (bend-twist coupling)
```

Tip slope (elliptic lift distribution, exact closed-form):

```
theta_tip = W_semi * L^2 / (8 * EI)            [rad]
```

The factor 1/8 comes from integrating q(y) = q_0sqrt(1-(y/L)^2) through the
Euler-Bernoulli cantilever:

```
theta_tip = (1/EI) integral_0L integral_y^L q(s)(s-y) ds dy  =  q_0 * L^3 * pi / (32 * EI)
```

with q_0 = 4W/(piL).

Total tip washout (sweep geometry + bend-twist coupling):

```
Deltaalpha_tip = -theta_tip * (tan Lambda + EK/GJ)         [rad]
```

All three stiffnesses (EI, GJ, EK) are differentiable with respect to ply
thicknesses `t` (through A_1_1 and D_6_6) and ply angles `theta` (through D_1_6 and
the Q-invariant form). IPOPT therefore receives exact Jacobians for the
aeroelastic constraint from CasADi's AD  --  the same as for Tsai-Wu and buckling.

### Physical interpretation

| Laminate type | D_1_6 | Washout mechanism |
|---|---|---|
| Balanced `[0/+/-45/90]s` | 0 | Geometric only  --  sweep + EI compliance |
| Unbalanced (free angles) | != 0 | Sweep + EK/GJ augmentation |

For a swept-back wing, EK > 0 (positive D_1_6) **augments** the geometric
washout. The optimizer can therefore achieve the required Deltaalpha_tip at **higher**
EI (thicker, stronger panels) than the geometry-only case, reducing the mass
penalty from the aeroelastic constraint.

The `demo_aeroelastic_tailoring.py` script shows the three-case comparison:
strength-only baseline (+0%), geometry washout (+33%), bend-twist coupled (+18%).

### Data flow

```
Flight condition + WingGeometry
         |
         v
optimize_laminate_aeroelastic(N, M, mat, angles, wing, n_load, relief_min)
         |
         |--- Tsai-Wu constraints          (same as optimize_laminate)
         |--- Buckling constraint          (same as optimize_laminate)
         |
         |--- A_1_1, D_1_6, D_6_6  <- _build_ABD_symmetric(t, Q_bar(theta))  [CasADi]
         |--- EI, GJ, EK      <- stiffness expressions               [CasADi]
         |--- theta_tip           <- W_semi*L^2/(8*EI)                    [CasADi]
         +--- Deltaalpha_tip <= -relief_min                                   [CasADi]
                  |
                  +--- opti.solve()  ->  AeroelasticOptimizationResult
```

---

## Sensitivity analysis

The parametric sensitivity of optimal mass to any flight parameter is available
analytically from the solved KKT conditions  --  one solve yields N sensitivities:

```
d(m*)/d(n_load)   --  load factor sensitivity
d(m*)/d(Mach)     --  Mach sensitivity
```

`demo_sensitivity.py` builds explicit sweep curves across n_load in [1.5, 4.0]g
and Mach in [1.4, 4.0] and annotates the slope at the nominal design point.
Normalised log-log sensitivities at the nominal point (n=2.5g, M=2.4, Alt=25.9km):

| Parameter | d(ln m*)/d(ln p) | Interpretation |
|---|---|---|
| n_load | 0.40 | Doubling load factor -> +33% mass |
| Mach | 0.16 | Doubling Mach -> +12% mass |

Load factor dominates; Mach matters primarily above M3.5 where dynamic
pressure growth accelerates.

---

## References

| | |
|---|---|
| CLT, ABD | Jones  --  *Mechanics of Composite Materials* (1999) Ch. 4 |
| Q invariant form | Kassapoglou  --  *Design and Analysis of Composite Structures* (2013) Sec.2.4 |
| Tsai-Wu | Tsai & Wu  --  *J. Composite Materials* 5(1), 1971 |
| Buckling (compression) | Timoshenko & Gere  --  *Theory of Elastic Stability* (1961) Ch. 9 |
| Buckling (shear, Seydel) | ESDU 02.03.11 |
| Buckling (interaction) | Whitney  --  *Structural Analysis of Laminated Anisotropic Plates* (1987) |
| Ackeret pressure | Ackeret  --  *ZAMM* 5(1), 1925 |
