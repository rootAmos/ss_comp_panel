# Modeling Assumptions and Fidelity Limitations

*A critical self-assessment of the composite panel MDO toolkit. Every assumption documented
here represents a deliberate trade between fidelity and tractability. Engineers using this
tool for real hardware must understand exactly where the physics end and the approximations
begin. The goal of this document is to make that boundary unmistakable.*

---

## Table of Contents

1. [Aerodynamic Load Model](#1-aerodynamic-load-model)
2. [Spanwise Load Distribution — The Elliptic Assumption](#2-spanwise-load-distribution--the-elliptic-assumption)
3. [Structural Panel Model (Classical Laminate Theory)](#3-structural-panel-model-classical-laminate-theory)
4. [Failure Criteria](#4-failure-criteria)
5. [Buckling Analysis](#5-buckling-analysis)
6. [Aeroelastic Coupling](#6-aeroelastic-coupling)
7. [Thermal Model and Aerodynamic Heating](#7-thermal-model-and-aerodynamic-heating)
8. [Trim and Lift Curve Slope](#8-trim-and-lift-curve-slope)
9. [Optimization Formulation](#9-optimization-formulation)
10. [Atmosphere Model](#10-atmosphere-model)
11. [Material Model](#11-material-model)
12. [Summary: When to Trust This Tool and When Not To](#12-summary-when-to-trust-this-tool-and-when-not-to)

---

## 1. Aerodynamic Load Model

### 1.1 Linearized Supersonic Theory (Ackeret, M > 1.15)

The pressure coefficient on the windward surface is:

```
ΔCp = 4α / √(M² − 1)
```

This is the Ackeret (1925) first-order thin-airfoil result. It assumes:

- **Small perturbation**: sin α ≈ α, cos α ≈ 1. Valid for α ≲ 10°. At α = 15° the error
  in sin α is ~4%; at α = 20° it exceeds 6%. For a fighter pulling 9g in a loaded maneuver,
  the panel AoA including aeroelastic twist and local incidence can easily exceed 10°.
- **Flat plate with zero thickness**: No leading-edge suction, no wave drag from camber or
  thickness. A real airfoil has a finite thickness ratio (t/c ~ 0.03–0.06 for supersonic
  sections); the thickness contribution to wave drag and chordwise pressure gradient is
  entirely absent here.
- **2-D flow**: No spanwise pressure gradient, no tip effects, no vortex sheet shed from
  the leading edge at finite sweep. The panel is treated as an infinite swept slab.
- **Leeward (upper) surface uses expansion suction** `Cp = −2α / √(M² − 1)`. This is
  physically correct for attached flow but will be wrong once the expansion produces a
  large suction that triggers separation or leads to a detached shock on the windward side.
  No mechanism exists in this code to detect flow separation.

**Validity range**: 1.15 < M < ~4 for α < 10°. Beyond M ~ 4–5, nonlinear effects and real-gas
corrections (see §1.4) become important even at small angles.

### 1.2 Transonic Blend (0.85 < M < 1.15)

The code linearly interpolates pressure coefficients between the Prandtl-Glauert subsonic
value at M = 0.85 and the Ackeret supersonic value at M = 1.15. This is purely a numerical
convenience to avoid the square-root singularity at M = 1. It has **no physical basis
whatsoever**. The transonic regime is governed by the full nonlinear transonic small-disturbance
(TSD) equation or the Euler equations. Mixed subsonic-supersonic flow, embedded shocks,
shock-induced separation, and buffet are completely absent. **Do not use this tool in the
transonic regime for any load-bearing structural sizing.** The blend exists only to keep the
optimizer numerically well-behaved when Mach number is a design variable sweeping across M = 1.

### 1.3 Subsonic Prandtl-Glauert (M ≤ 0.85)

```
ΔCp = 4α / √(1 − M²)
```

Same flat-plate, small-angle linearization as Ackeret, now in the subsonic compressibility
correction sense. Ignores camber, viscosity, and stall. The M = 0.85 cutoff is conservative;
compressibility corrections begin to fail for M ≳ 0.7 near the leading edge of real sections
due to local supersonic patches.

### 1.4 Modified Newtonian Impact (M > 5)

For hypersonic speeds the code uses:

```
Cp_windward = Cp_max · sin²(α)
Cp_leeward  = 0   (shadow assumption)
```

where `Cp_max` is computed from the Rayleigh pitot formula for normal-shock stagnation.

This is a reasonable upper-surface loading estimate for blunt-body or panel flows at high
incidence. However:

- **γ = 1.4 is assumed constant (perfect gas)**. Above M ~ 5 at sea level (and certainly
  above M ~ 7 at altitude), air dissociates. Real-gas effects reduce the effective γ toward
  ~1.2–1.3, changing stagnation pressure and surface pressure meaningfully. The errors in
  Cp_max can reach 15–25% at M = 10.
- **Leeward surface zero suction**: Real hypersonic vehicles have small but non-zero leeward
  pressures from expansion around edges. For panel shear and torsion this matters.
- **No viscous interaction**: Hypersonic boundary layers are thick relative to wing thickness.
  The displacement effect and the strong viscous-inviscid interaction (Bertram's parameter
  χ = M²√C_f) alter surface pressures substantially on slender configurations.

### 1.5 Oblique Shock (Intermediate Mach)

The θ-β-M relation is solved by Newton-Raphson to get the exact attached-shock pressure ratio.
The weak-shock branch is selected. This is correct as far as it goes, but:

- Assumes **attached shock at the leading edge**. Above the detachment angle for a given M,
  no attached shock exists and the flow goes through a detached bow shock. The code does not
  check for detachment and will produce physically meaningless pressures.
- **Sweep is not accounted for in the oblique shock solution**. A swept leading edge sees
  an effective Mach normal component `M_n = M cos(Λ_LE)`. For highly swept delta wings at
  supersonic speeds this correction is not negligible.

### 1.6 Chordwise Pressure Distribution

The entire chordwise pressure field is collapsed to a single resultant. The code computes:

```
Nyy = −Δp · c / 2
Mxx = Δp · c² / 8
```

This is the **simply-supported beam analogy**: uniform pressure over a plate supported at
its leading and trailing edges (ribs or spar caps), producing a maximum bending moment at
mid-chord of `p·L²/8`. The actual chordwise pressure distribution from Ackeret theory is
uniform only for a flat plate at constant AoA. Camber, thickness, and local shock
interactions create non-uniform distributions that shift the load resultant. For a cambered
section the aerodynamic center is at the quarter-chord; the simply-supported analogy implicitly
assumes the load is equally distributed between two edges, which is a reasonable first pass
but not an accurate aerodynamic loading.

---

## 2. Spanwise Load Distribution

### 2.1 Implemented model: Mach-regime-aware blend

**`wing_panel_loads(distribution='auto')`** uses a blend of two analytically justified
distributions, weighted by the supersonic leading-edge Mach parameter K = M·cos(Λ_LE):

```
l(y) = (1 − w) · l_elliptic(y)  +  w · l_chord(y)

l_elliptic(y) = l₀ · √(1 − (y/b)²)          Prandtl, 1918
l_chord(y)    = (L/2) · c(y) / S_half         chord-proportional (strip theory)

w = 0.5                         for M ≤ 0.85  (Schrenk approximation)
w = 0.5 + 0.5·clip((K−0.5)/1.5) for M ≥ 1.15 (ramps to 1.0 at K = 2)
```

`distribution='elliptic'` forces the original purely elliptic assumption for
back-compatibility or conservative bounding.  See `spanwise_lift_distribution()`
for a plotting function that returns (y, l) arrays.

### 2.2 Physical basis for each limiting distribution

**Elliptic (w = 0)**: the Prandtl (1918) minimum-induced-drag solution; exact only for
an elliptic planform in incompressible, high-AR, unswept flow.  For a trapezoidal wing
it overestimates outboard loading relative to the geometric (chord) distribution.

**Chord-proportional (w = 1)**: the Ackeret strip-theory limit.  Each spanwise station
is aerodynamically independent; lift per unit span equals (4α/β)·q·c(y).  This is the
exact result for a flat wing with a **fully supersonic leading edge** (K = M·cos(Λ_LE) > 1),
where no spanwise signal can propagate upstream of the Mach cone from each leading-edge
point (Evvard 1950, NACA Report 951).

**Schrenk (w = 0.5)**: the 50/50 blend (NACA TM 948, 1940).  Validated against panel
methods to within ~5% for AR > 4, Λ < 35°.  Used throughout the subsonic regime.

### 2.3 Why the pure elliptic assumption was wrong for this tool

The elliptic distribution is the incompressible, high-AR lifting-line limit. This tool
targets supersonic and hypersonic wings, which violate every premise of that limit:

1. **AR ~ 2–4**: Lifting-line theory requires AR ≫ 1; errors exceed 15% below AR = 4.
2. **Supersonic leading edges**: Once K > 1, each strip decouples. The elliptic
   coupling kernel becomes physically meaningless—there is no upstream influence at
   all from the tip for a supersonic LE.
3. **Mach-cone tip unloading**: The region of the wing within the tip Mach cone has
   reduced loading relative to the inboard panel, producing a distribution that is more
   inboard-concentrated than an ellipse, not less.
4. **Taper**: A tapered trapezoidal wing with λ ~ 0.2–0.3 has a small tip chord. The
   geometric (chord-proportional) distribution correctly loads the root heavily and the
   tip lightly; the elliptic distribution ignores this entirely.

For a typical fighter wing (λ=0.25, Λ_LE=45°, M=2.0, K=1.41), the auto blend gives
**w ≈ 0.80**: 80% chord-proportional, 20% elliptic.  The resulting root bending moment
is ~6–8% lower than the pure elliptic result.  This is not a trivial correction—it
directly sets the skin thickness and mass of the heaviest (root) panels.

### 2.4 Remaining limitations of the implemented model

The blend captures the dominant physics but still omits:

- **Tip Mach cone geometry**: The exact supersonic loading in the tip region requires
  tracking the intersection of the wing planform with the Mach cone from the tip trailing
  edge. The blend uses only the scalar K parameter and cannot capture the chordwise
  variation of this effect near the tip.
- **Subsonic leading-edge suction**: For K < 1 (subsonic LE), the leading edge carries a
  suction force that redistributes load toward the LE. This appears in the Evvard integral
  as an arcsin singularity at the leading edge and is absent from both limiting distributions
  used here.
- **Cranked or delta planforms**: The trapezoidal chord law `c(y) = c_root·(1−(1−λ)η)` is
  exact only for a single-panel trapezoidal wing. Cranked planforms (inner wing + outer
  panel) require separate treatment of each panel.
- **Evvard-Krasner conical flow** (the next-fidelity upgrade): For a delta wing with a
  supersonic leading edge, Evvard (1950) gives the exact closed-form pressure distribution
  as an integral over the wing leading edge. The spanwise lift per unit span involves the
  conical similarity variable ξ = β·y/x and the distribution takes the form:
  ```
  l(η) ∝ c(η) · [1 − F(K, η/b)]
  ```
  where F is an elliptic-integral correction that reduces to zero for K ≫ 1 (pure strip)
  and to an arcsin expression for K < 1 (subsonic LE). Implementing Evvard exactly would
  require tracking the full (x, y) planform geometry and is deferred to a future version.

### 2.5 Sweep correction applied to Nxx

The code applies a `cos²(Λ_LE)` factor to convert bending moment to skin axial stress,
projecting the bending moment along the structural spar axis. This is exact for a straight
elastic axis but becomes inaccurate for delta wings (Λ ~ 60–70°) where the oblique-beam
analogy breaks down and a full swept-beam stiffness matrix is needed.

---

## 3. Structural Panel Model (Classical Laminate Theory)

### 3.1 Kinematic Assumptions (Kirchhoff-Love)

The entire structural model rests on Classical Laminate Theory (CLT). CLT assumes:

```
ε(z) = ε₀ + z · κ
```

where ε₀ are midplane strains and κ are curvatures. This is the direct extenstion of
Kirchhoff's plate hypothesis to a laminated medium. Consequences:

- **Plane stress**: σ_z = τ_xz = τ_yz = 0 through the thickness. For thin panels (h/a ≲ 1/10)
  this is an excellent approximation. For thick panels or near-edge regions (within ~1 ply
  thickness of a free edge), the through-thickness stresses are non-negligible and CLT is
  quantitatively wrong.
- **No transverse shear deformation** (Timoshenko-Mindlin correction omitted). The shear
  stiffness is assumed infinite. For typical AS/IM7 carbon-epoxy panels with t/c < 0.02 and
  panel span-to-thickness > 50, this error is small (<3%). For thick sandwich panels or
  foam-core constructions it is not.
- **No geometric nonlinearity**: The curvature is taken as κ = w_xx (small deflection). Once
  transverse deflection w exceeds ~0.3 times the panel thickness h, the membrane-bending
  coupling (von Kármán terms) becomes important and CLT in its linear form underestimates
  stiffness (the panel stiffens in tension due to in-plane stretching). The postbuckling
  reserve strength—which can be substantial for compression-loaded composite panels—is
  entirely absent.

### 3.2 ABD Matrix and Laminate Symmetry

The code enforces **symmetric laminates** (plies mirrored about the mid-plane), which forces
B = 0. This is the correct default for flight structure to avoid thermally-induced warping
during cure and bending-extension coupling under mechanical load. If a user attempts to model
an asymmetric laminate, the optimizer result is valid only within the CLT framework; the actual
cure-distortion and resulting residual stress state require a nonlinear cure-simulation.

**Balanced laminates** (+θ/−θ pairs) zero out A_16 and A_26. The optimizer optionally enforces
this. An unbalanced laminate will shear when pulled in tension, which is usually undesirable
but sometimes exploited for aeroelastic tailoring. The code does not prevent unbalanced designs;
it merely warns when D_16 and D_26 are large relative to √(D_11 · D_22).

### 3.3 Panel Boundary Conditions

All buckling and bending calculations assume **simply-supported edges on all four sides**.
Real panels are attached to ribs, stringers, and spar caps with finite rotational stiffness.
A fully clamped edge raises the critical buckling load by a factor of ~4 relative to simply
supported for uniaxial compression; a real attachment is between these extremes. The simply
supported assumption is **conservative for buckling** (unconservative for interlaminar failure
near clamped roots). The code does not parameterize edge restraint and provides no way to
account for partial fixity.

### 3.4 Rib and Stringer Spacing

The panel dimensions `a × b` (spanwise × chordwise) are user inputs representing the
rib and stringer pitch. The code has no substructure model—ribs and stringers are treated
as rigid, perfectly-clamped supports that transfer load into the next bay. In reality,
rib flexibility reduces the effective spanwise support stiffness, the stringer eccentricity
introduces additional bending moments, and fastener flexibility at attachments creates load
introduction effects that CLT does not capture.

---

## 4. Failure Criteria

All four implemented criteria operate at the **first-ply failure (FPF)** level. This means
failure is declared at the load at which the most heavily loaded ply first reaches its
criterion. No redistribution, no progressive damage, no stiffness knockdown. For preliminary
design this is standard practice; for final sizing the continued-loading capability (last-ply
failure, or limit-load capability with damage) must be assessed separately.

### 4.1 Tsai-Wu (primary design criterion)

The Tsai-Wu polynomial failure criterion accounts for tension-compression asymmetry in both
fibre and matrix directions. The interaction coefficient F₁₂* is set to the conservative
Tsai (1971) estimate:

```
F₁₂* = −0.5 / √(F₁ₜ · F₁c · F₂ₜ · F₂c)
```

This is the **most uncertain parameter in the entire failure model**. F₁₂* requires
off-axis biaxial testing to determine experimentally. The Tsai value of −0.5 is an educated
bound, not a material constant. Published experimental values for IM7/8552 range from −0.3
to −0.6 depending on the test method. The sensitivity of the predicted RF to F₁₂* is moderate
for fibre-dominated layups but can be significant for ±45° shear panels.

### 4.2 Hashin (progressive/mode-specific checks)

The Hashin criterion separates fibre-dominated and matrix-dominated failure modes, which
improves physical insight. However, for fibre compression, the criterion uses:

```
FI_fibre_comp = (σ₁ / F₁c)²
```

with no shear coupling. This follows Hashin (1980) original; the more recent Hashin-Rotem
(1973) and Puck (1996) formulations suggest shear coupling even in fibre compression mode.
For panels loaded in combined axial compression and shear (common on torsion boxes), this
omission may be unconservative.

### 4.3 Stress Recovery and Interlaminar Failure

Stresses are recovered at ply midplanes only. CLT stresses are constant through each ply
(zeroth-order recovery). The interlaminar shear and peel stresses at ply interfaces—which
drive delamination, the predominant failure mode in impact-damaged composite structure—are
**not computed**. Free-edge delamination at the ply drop-offs, which is critical at panel
edges near fastener holes, is entirely outside the scope of this model.

---

## 5. Buckling Analysis

### 5.1 Rayleigh-Ritz Plate Buckling (Timoshenko/Whitney)

The critical buckling load for orthotropic simply-supported rectangular panels is:

```
Nxx_cr = (π² / b²) · min_m [ D₁₁(mb/a)² + 2(D₁₂ + 2D₆₆) + D₂₂(a/(mb))² ]
```

This is the classical result. It is **exact** for orthotropic plates with D₁₆ = D₂₆ = 0
(balanced symmetric laminates) under uniform end compression with simply-supported edges.
Its assumptions:

- **Linear bifurcation buckling**: The prebuckling state is assumed uniform compression
  (no prebuckling bending), and the critical load is found as a linear eigenvalue. There is
  no knock-down factor for geometric imperfections, which are the dominant driver of the
  difference between experimental and classical buckling loads for compression-loaded shells
  and panels. A knock-down of 10–30% is typical for composite panels depending on layup
  and manufacturing quality.
- **D₁₆ = D₂₆ = 0 assumed for formula validity**. The code checks this and warns when
  unbalanced layups produce significant coupling. However, the warning is advisory; the
  formula is applied regardless.
- **Fixed mode number m = 1 in the CasADi (optimization) path**: The numpy path optimizes
  over m = 1..8. The optimizer uses m = 1 to maintain a smooth, differentiable constraint.
  For panels with a/b > 2, the true critical mode may have m = 2 or 3, meaning the optimizer's
  buckling constraint is **non-conservative** (it overestimates the critical load) by up to
  ~10–15% for long, narrow panels.
- **No postbuckling**: Composite panels often carry 1.5–3× their buckling load before final
  failure. The postbuckling regime is not modeled.

### 5.2 Shear Buckling (Seydel / ESDU)

The shear buckling coefficient:

```
ks = 8.125 + 5.045 / η,    η = (D₁₂ + 2D₆₆) / √(D₁₁ · D₂₂)
```

is enforced with a floor η ≥ 0.5. This is necessary for numerical stability but also correct
physically (highly anisotropic layups have η < 0.5 only in pathological cases). The Seydel
formula is an empirical fit to the exact orthotropic solution and is accurate to ~5% over
the range 0.5 < η < 5.

### 5.3 Combined Loading (Whitney Interaction)

```
Rₓ + Rᵧ + Rs² = 1
```

This is the Whitney (1987) extension of the Stowell interaction formula to orthotropic plates.
It is a reasonable approximation for panels with moderate coupling but has not been validated
extensively for highly anisotropic laminates or for combined biaxial compression and shear.
The interaction formula is **not conservative** in all quadrants of load space; physical test
data should be used to validate the interaction surface for the specific laminate.

---

## 6. Aeroelastic Coupling

### 6.1 Euler-Bernoulli Beam with Torsional Relief

The spanwise bending is modeled as a root-clamped cantilever beam. The bending stiffness is
derived from the CLT extensional modulus:

```
EI(y) = 2 · E_eff · b_box · (h_box / 2)²
```

where `b_box = 0.5 · chord` (50% chord box, hardcoded) and `h_box = (t/c) · chord`. This is
the **I-beam flange analogy**: a rectangular torsion box with bending moment carried entirely
by the top and bottom flanges, neglecting web contribution. For a well-designed wing box
with spar web panels the flange dominates, but the 50% chord fraction is a rough approximation
and should be replaced with the actual structural box width at each station.

**Shear deformation (Timoshenko) is neglected**. For a wing with span-to-depth ratio > 15
(typical for fighter configurations) shear deformation contributes < 5% to tip deflection.
For short, thick-section wings this may not hold.

### 6.2 Washout Twist Approximation

The aeroelastic twist relief (washout) is computed as:

```
Δα(y) = −θ(y) · tan(Λ_LE)
```

where θ(y) is the beam bending slope. This is the small-angle geometric approximation for
how bending of a swept beam produces a change in local angle of attack. The sign convention
is that upward bending of a forward-swept wing produces wash-in (more lift) and of a
rearward-swept wing produces washout (less lift). The formula is linearized in both the
deflection and the sweep angle and becomes inaccurate for large deflections or sweeps
approaching 60–70° (typical of delta wings).

**Torsional stiffness is not modeled separately.** The wing is treated as a pure bending beam
with no torsional degree of freedom. For wings with significant pitch-roll coupling (low GJ
relative to EI), the torsional aeroelastic response—divergence in particular—requires a
full bending-torsion beam with separate GJ and EA-AC offset treatment. The Bredt shear-flow
correction for pitching moment is included in the load computation but does not feed back
into a structural torsion response.

### 6.3 Iterative Convergence

The load-deformation loop iterates up to 10 times with a 0.01° twist convergence tolerance.
There is no formal proof of convergence, and no divergence detection. For configurations
near aeroelastic divergence (high dynamic pressure, soft structure), the iteration may
oscillate or fail to converge within 10 steps without warning. The speed index
`q_∞ · CLα · b² / (GJ)` should be checked against the divergence boundary before applying
this tool near the transonic or low-altitude high-speed corner of the flight envelope.

---

## 7. Thermal Model and Aerodynamic Heating

### 7.1 Eckert Reference Temperature Method

Aerodynamic heating is computed via the Eckert (1955) reference-temperature method for
turbulent flat-plate flow:

```
St* = 0.0296 · Re_x*^(−0.2) · Pr*^(−2/3)
```

evaluated at the reference temperature:

```
T* = 0.5(T_∞ + T_wall) + 0.22(T_aw − T_∞)
```

This is a well-validated correlation for attached turbulent boundary layers over flat plates
in high-speed flow. Its limitations:

- **Assumes turbulent flow from the leading edge**. For low Reynolds numbers or polished
  surfaces, the flow may be transitional or laminar over significant portions of the panel.
  Laminar heat transfer rates are 3–5× lower than turbulent for the same Re.
- **Flat plate, zero pressure gradient**. Real wing surfaces have favorable (accelerating)
  pressure gradients near the leading edge and adverse gradients toward the trailing edge.
  Favorable gradients reduce heat transfer; adverse gradients increase it.
- **No shock-layer or shock-boundary-layer interaction**. On hypersonic vehicles, a reattaching
  flow at a concave-corner shock impingement can produce local heat fluxes 5–10× the flat-plate
  value. Panel leading edges and root attachments are particularly susceptible.
- **γ = 1.4 (perfect gas) and constant Prandtl number Pr = 0.71**. Above M ~ 7–8 at altitude,
  vibrational excitation and dissociation reduce γ toward ~1.15–1.2 and change the recovery
  factor and stagnation enthalpy substantially. The code is **not physically valid for true
  hypersonic entry conditions**.

### 7.2 Cure-Residual Stresses

Thermal stresses are computed relative to the **stress-free temperature = cure temperature
T_cure = 177°C (450 K)**. This is correct for the residual stress state locked in during
autoclave cure assuming the laminate is stress-free at the cure temperature. In practice:

- Matrix creep and relaxation during the cure cycle mean the effective stress-free temperature
  is somewhat below the peak cure temperature (~160–170°C is more common in practice).
- Multiple cure cycles (e.g., secondary bonding or repair patches) create a layered residual
  stress state that is not modeled.
- The large transverse CTE (α₂ ~ 28.8 µ/K) relative to the fiber direction (α₁ ~ 0.3 µ/K)
  creates significant **matrix microcracking** in cross-plied laminates when cooled from cure.
  These microcracks reduce the effective transverse stiffness E₂ and strength F₂ₜ. No
  degradation model is applied.

### 7.3 Through-Thickness Temperature Distribution

A **linear** gradient through the panel thickness is assumed. For a thin carbon-epoxy skin
(h ~ 2–6 mm) with high in-plane thermal conductivity and moderate through-thickness
conductivity, this is a reasonable approximation for quasi-steady heating. During rapid
transient heating (reentry pulse), the actual gradient may be nonlinear due to the finite
thermal diffusivity of the composite. Thermal stresses during transients can exceed those at
steady state.

---

## 8. Trim and Lift Curve Slope

### 8.1 Lift-Curve Slope Corrections

**Subsonic**: The Helmbold finite-wing correction is applied to the Prandtl-Glauert 2D slope:

```
CLα = (2π/β) / (1 + (2π/β) / (πAR·e))
```

This is the incompressible low-AR correction by Helmbold (1942), compressibility-corrected
via Prandtl-Glauert. It is reasonably accurate for straight and moderately swept wings
(Λ ≲ 35°) at moderate AR. For highly swept delta wings it underestimates the leading-edge
vortex contribution to lift.

**Supersonic**: The Jones (1947) supersonic finite-wing correction:

```
CLα = (4/β) / √(1 + (4/(βπAR))²)
```

is applied. This is an asymptotic result from slender-body theory blended into the Ackeret
2D limit. It is accurate for slender (low-AR) supersonic wings but underestimates tip-loss
corrections for moderate-AR configurations.

The trim calculation is strictly **linear lift theory**. No stall model, no vortex-induced
nonlinearity, no buffet boundary. For angle-of-attack requirements beyond ~12–15° the
linear lift slope will diverge from the true CL significantly.

### 8.2 Oswald Efficiency Factor

The Oswald factor is estimated from the Nita-Scholz (2012) correlation as a function of
taper ratio and sweep. This is a statistical regression; the uncertainty on e for an
individual design can be ±15–20%. The Oswald factor appears in the denominator of the
induced drag and in the Helmbold correction, so errors here propagate directly to trim AoA
and thus to all panel load levels.

---

## 9. Optimization Formulation

### 9.1 Continuous Ply Thickness Variables

Ply thicknesses `t_k` are treated as continuous real variables. In reality, composite
laminates are built from discrete plies with fixed ply thickness (CPT ~ 0.125 mm for standard
prepreg). A design calling for `t_k = 0.31 mm` will be rounded to 2 or 3 plies in production,
potentially violating the optimized RF constraints by 20–30%. The optimizer output should
be treated as a **minimum-mass lower bound**; the discrete ply-count design will always be
heavier.

### 9.2 First-Ply Failure Constraint

The Tsai-Wu RF constraint is applied at the FPF level. There is no constraint on fatigue,
damage tolerance, minimum ply count rules, or ply percentage bounds (e.g., minimum 10% 0°
plies for damage arrestment, maximum 50% 0° plies for balanced stiffness). These are
standard aerospace design rules that would typically be added as side constraints. Their
absence means the optimizer can produce all-0° or all-45° laminates that are structurally
efficient against the analyzed load case but fragile against out-of-plane impact or secondary
bending.

### 9.3 Mode Shape Fixed in Buckling Constraint

For differentiability, the optimization uses the fixed-mode (m = 1) buckling formula. This
can overestimate the critical buckling load for long panels (see §5.1). The resulting
optimization may find designs that appear to satisfy the buckling constraint but would fail
a detailed analysis with mode optimization.

### 9.4 Panel Independence (No Structural Coupling Between Stations)

The wing optimizer sizes each spanwise station independently. The optimized laminate at
η = 0.5 has no knowledge of the laminate at η = 0.4 or η = 0.6. In a real wing:

- **Laminate continuity**: Plies must run continuously from root to tip (or step down at
  defined ply-drop locations). Abrupt laminate transitions create eccentricity moments and
  peel stresses at the ply drop that are not captured here.
- **Load redistribution**: In a statically indeterminate structure, the stiffness distribution
  changes the load distribution. This tool uses a predetermined load (elliptic distribution)
  and sizes the structure to it without iterating the structural stiffness back into the
  load computation (except for the simplified aeroelastic correction).

### 9.5 Global Minimum Not Guaranteed

The CasADi/IPOPT gradient-based optimizer finds a **local** minimum. The structural
optimization problem with multiple coupled failure constraints and nonlinear trigonometric
terms in the ply angle variables is non-convex. The solution depends on the initial guess.
A multi-start strategy is advisable for angle-optimization problems; for thickness-only
optimization the problem is convex in the log-thickness space and the local minimum is
typically global.

---

## 10. Atmosphere Model

The ISA model is implemented as a two-layer model: troposphere (0–11 km, lapse rate
−6.5 K/km) and isothermal stratosphere (11–20 km). This is accurate for the standard day
by definition. It does not model:

- **Hot day, cold day, or tropical day** atmospheric profiles (MIL-HDBK-310 envelopes)
- **Altitudes above 20 km** (relevant for sustained Mach 5+ cruise or boost-glide trajectories)
- **Chemical composition changes** in the mesosphere and above, where molecular weight and γ
  deviate from sea-level values

For thermal analysis at high Mach numbers, the choice of standard-day vs. hot-day conditions
can change the aerodynamic heating by 10–15% due to the difference in freestream density
and temperature.

---

## 11. Material Model

### 11.1 Plane Stress, Linear Elastic, No Rate Dependence

The ply is modeled as a linear elastic orthotropic continuum under plane stress. No:

- **Fiber waviness or misalignment**: Manufacturing processes produce fiber paths that deviate
  from the nominal by 1–3°. This reduces compression strength by 5–15% and is not reflected
  in the nominal F₁c allowable unless the user applies a knock-down.
- **Matrix nonlinearity**: Epoxy matrices exhibit nonlinear shear response (softening) at
  shear strains above ~1%. The linear G₁₂ assumption overestimates shear stiffness at high
  shear load levels.
- **Temperature-dependent properties**: E₁, E₂, G₁₂, and the strengths F₁ₜ, F₁c, F₂ₜ, F₂c,
  F₁₂ all degrade with temperature. For panels operating above ~120°C (well within the range
  for sustained supersonic flight), the matrix-dominated properties E₂ and F₂ₜ can degrade
  by 30–50% of their room-temperature values. The hardcoded IM7/8552 properties are
  room-temperature, ambient-humidity values.
- **Moisture absorption**: Absorbed moisture (wet layup conditions) further reduces T_g and
  degrades matrix properties. The **hot-wet knockdown**—typically applied as a
  temperature-moisture combined environment (e.g., 180°F/0.5% moisture for certification)—
  is not incorporated.
- **Fatigue**: No S-N curve or damage accumulation model.

### 11.2 IM7/8552 Default Properties

The default material is IM7/8552 carbon-epoxy prepreg, which is appropriate for modern
supersonic and hypersonic structural applications up to ~180°C continuous service temperature.
The properties are based on published test data at room temperature. For use above ~120°C,
**user-supplied hot-wet-corrected allowables must replace these values explicitly**.

---

## 12. Summary: 

### Reliable (within ±15% of reference-quality analysis)

| Task | Confidence | Notes |
|------|-----------|-------|
| Subsonic/supersonic panel load estimation for M < 4, α < 8° | Moderate | Ackeret linearization holds |
| CLT ABD matrix and effective moduli | High | Standard result, well-validated |
| Tsai-Wu FPF for in-plane dominated layups | Moderate-High | F₁₂* uncertainty is main risk |
| Uniaxial buckling of simply-supported orthotropic panel | Moderate-High | m=1 fix in optimizer; numpy path is exact |
| Preliminary wing skin mass estimation | Moderate | ±20–30% vs. detailed sizing expected |
| Aeroelastic twist relief for moderate sweep (Λ < 40°), M < 2 | Moderate | Euler-Bernoulli shear neglected |
| Cure-residual thermal stress at ambient conditions | Moderate | Linear gradient adequate for thin skins |
