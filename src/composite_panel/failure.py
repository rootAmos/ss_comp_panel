"""
composite_panel.failure
-----------------------
Ply-level failure criteria for composite laminates.

BACKGROUND — WHY PLY-LEVEL FAILURE?
=====================================
Unlike isotropic metals, composite materials fail in fundamentally different
modes depending on the direction of loading:

  - Fibre direction (1-axis): very strong in tension (fibre fracture),
    lower in compression (fibre microbuckling / kinking)
  - Transverse direction (2-axis): matrix-dominated, much weaker — especially
    in tension (matrix cracking, delamination onset)
  - Shear (12-plane): matrix/interface dominated

A single isotropic von Mises check would be wildly non-conservative for a ply
loaded transversely in tension (F2t ~ 62 MPa vs F1t ~ 2326 MPa for IM7/8552).

All criteria here operate on PRINCIPAL PLY AXES stresses [σ1, σ2, τ12]
obtained by rotating the laminate-frame stresses via the T matrix (see ply.py).

CRITERIA IMPLEMENTED
====================

1. Maximum Stress  (conservative, non-interactive)
   Independent limits on each component.  Simple, easy to understand,
   identifies the governing failure mode directly.  Does NOT capture
   stress interaction — can be overly conservative or unconservative.

2. Maximum Strain
   Same philosophy as max stress but in strain space.  Occasionally
   preferred for materials where failure is strain-limited (e.g. woven fabrics).

3. Tsai-Hill  (interactive, single-strength)
   Energy-based criterion derived from Hill's anisotropic yield criterion.
   Accounts for stress interaction but does not distinguish tension/compression
   strengths.  Symmetric in σ1 and -σ1.

4. Tsai-Wu  (interactive, dual-strength — INDUSTRY STANDARD)
   Tensor polynomial criterion.  Full form:
       F1·σ1 + F2·σ2 + F11·σ1² + F22·σ2² + F66·τ12² + 2F12*·σ1·σ2 = 1

   Unlike Tsai-Hill, it separately accounts for different tension/compression
   strengths via the linear terms (F1·σ1, F2·σ2).

   The interaction coefficient F12* is experimentally determined but often
   approximated as -0.5/√(F1t·F1c·F2t·F2c) — Tsai's recommendation when no
   biaxial test data is available.

   RF is solved from the quadratic in load-scale factor λ:
       (F11·σ1² + F22·σ2² + F66·τ12²)·λ² + (F1·σ1 + F2·σ2)·λ - 1 = 0
   →  λ = RF = [-b + √(b² + 4a)] / (2a)

RESERVE FACTOR DEFINITION
==========================
    RF = (allowable load level) / (applied load level)

    RF > 1.0 → safe (applied load is below allowable)
    RF = 1.0 → on the limit (first-ply failure)
    RF < 1.0 → failed (applied load exceeds allowable)

This is opposite to a Failure Index (FI = 1/RF), but is the more intuitive
convention for structural sizing — "how many times can I scale up the load
before this ply fails?"

FIRST-PLY FAILURE vs PROGRESSIVE DAMAGE
========================================
This implementation predicts FIRST-PLY FAILURE (FPF) — the onset of damage
in the most critically loaded ply.  It does NOT model progressive damage
accumulation (ply degradation, stiffness knockdown, last-ply failure).

For preliminary and conceptual design sizing, FPF with Tsai-Wu and a design
reserve factor of ≥ 1.5 is standard practice and conservative enough to be
used without progressive damage analysis.

Reference:
    Kassapoglou, C. – Design and Analysis of Composite Structures
    (Wiley, 2013), Ch. 5

    Tsai, S.W. & Wu, E.M. – A General Theory of Strength for Anisotropic Materials
    Journal of Composite Materials, 5(1), 1971, pp. 58–80.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional
from .ply import PlyMaterial


@dataclass
class PlyFailureResult:
    """
    Failure assessment output for a single ply.

    Stores both the individual directional margins and the governing
    combined reserve factor.  Individual RFs help the designer understand
    WHICH failure mode is critical (fibre? matrix? shear?) — essential for
    deciding how to change the layup.

    Attributes
    ----------
    ply_index : int
        Position in the ply stack (0 = bottom ply).
    angle_deg : float
        Fibre angle of this ply [degrees].
    criterion : str
        Name of the failure criterion used.
    rf_1 : float
        Reserve factor in the fibre direction (σ1).
    rf_2 : float
        Reserve factor in the transverse direction (σ2).
    rf_12 : float
        Reserve factor in shear (τ12).
    rf_combined : float
        Combined reserve factor from the interaction criterion.
    rf : float
        Governing (minimum) reserve factor for this ply.
    failed : bool
        True if rf < 1.0 (first-ply failure).
    """
    ply_index:   int
    angle_deg:   float
    criterion:   str

    # Directional margins — useful for layup optimisation decisions
    rf_1:  float   # fibre-direction margin
    rf_2:  float   # transverse margin  (often critical for off-axis plies)
    rf_12: float   # shear margin

    rf_combined: float   # interaction index from Tsai-Wu / Tsai-Hill
    rf: float            # governing RF (what the designer acts on)
    failed: bool

    def __str__(self) -> str:
        status = "FAIL" if self.failed else "OK  "
        return (f"  [{status}] Ply {self.ply_index:2d}  θ={self.angle_deg:5.1f}°  "
                f"RF={self.rf:.4f}  (1:{self.rf_1:.3f}  2:{self.rf_2:.3f}  "
                f"12:{self.rf_12:.3f}  comb:{self.rf_combined:.4f})")


# ---------------------------------------------------------------------------
# Maximum Stress Criterion
# ---------------------------------------------------------------------------

def max_stress(mat: PlyMaterial,
               sig_12: np.ndarray,
               ply_index: int = 0,
               angle_deg: float = 0.0) -> PlyFailureResult:
    """
    Maximum stress failure criterion.

    Each stress component is checked independently against its relevant
    strength allowable.  No interaction between components.

    Decision logic:
      σ1 > 0  →  check against F1t  (tensile fibre failure)
      σ1 < 0  →  check against F1c  (compressive fibre failure / kinking)
      σ2 > 0  →  check against F2t  (matrix tensile cracking)
      σ2 < 0  →  check against F2c  (matrix compression)
      τ12     →  check against F12  (interlaminar shear, sign-independent)

    RF_i = allowable_i / |applied_i|
    Governing RF = min(RF_1, RF_2, RF_12)

    Parameters
    ----------
    mat : PlyMaterial
        Material with strength allowables.
    sig_12 : np.ndarray, shape (3,)
        Ply-axis stresses [σ1, σ2, τ12] in Pa.
    ply_index : int
        Ply index for result labelling.
    angle_deg : float
        Fibre angle for result labelling.

    Returns
    -------
    PlyFailureResult
    """
    s1, s2, s12 = sig_12

    # Select tension or compression allowable based on stress sign
    rf_1  = mat.F1t / s1       if s1  > 0 else mat.F1c / abs(s1)  if s1  < 0 else np.inf
    rf_2  = mat.F2t / s2       if s2  > 0 else mat.F2c / abs(s2)  if s2  < 0 else np.inf
    rf_12 = mat.F12 / abs(s12) if s12 != 0 else np.inf   # shear strength is symmetric

    rf = min(rf_1, rf_2, rf_12)
    return PlyFailureResult(ply_index, angle_deg, "MaxStress",
                            rf_1, rf_2, rf_12, rf_combined=rf,
                            rf=rf, failed=(rf < 1.0))


# ---------------------------------------------------------------------------
# Maximum Strain Criterion
# ---------------------------------------------------------------------------

def max_strain(mat: PlyMaterial,
               eps_12: np.ndarray,
               ply_index: int = 0,
               angle_deg: float = 0.0) -> PlyFailureResult:
    """
    Maximum strain failure criterion.

    Equivalent to max stress but in strain space.  Ultimate strains are
    derived from strengths and elastic moduli (linear-elastic assumption).

    Ultimate strains:
        ε1t_ult  = F1t / E1
        ε1c_ult  = F1c / E1
        ε2t_ult  = F2t / E2
        ε2c_ult  = F2c / E2
        γ12_ult  = F12 / G12

    This criterion can give different results from max stress when Poisson
    effects are significant (which they are for high-ν12 composites).

    Parameters
    ----------
    mat : PlyMaterial
    eps_12 : np.ndarray, shape (3,)
        Ply-axis engineering strains [ε1, ε2, γ12].
    ply_index, angle_deg : for labelling.

    Returns
    -------
    PlyFailureResult
    """
    e1, e2, g12 = eps_12

    # Convert strength allowables to ultimate strain allowables
    e1t_ult  = mat.F1t  / mat.E1
    e1c_ult  = mat.F1c  / mat.E1
    e2t_ult  = mat.F2t  / mat.E2
    e2c_ult  = mat.F2c  / mat.E2
    g12_ult  = mat.F12  / mat.G12

    rf_1  = e1t_ult  / e1    if e1  > 0 else e1c_ult  / abs(e1)  if e1  < 0 else np.inf
    rf_2  = e2t_ult  / e2    if e2  > 0 else e2c_ult  / abs(e2)  if e2  < 0 else np.inf
    rf_12 = g12_ult  / abs(g12) if g12 != 0 else np.inf

    rf = min(rf_1, rf_2, rf_12)
    return PlyFailureResult(ply_index, angle_deg, "MaxStrain",
                            rf_1, rf_2, rf_12, rf_combined=rf,
                            rf=rf, failed=(rf < 1.0))


# ---------------------------------------------------------------------------
# Tsai-Hill Criterion
# ---------------------------------------------------------------------------

def tsai_hill(mat: PlyMaterial,
              sig_12: np.ndarray,
              ply_index: int = 0,
              angle_deg: float = 0.0) -> PlyFailureResult:
    """
    Tsai-Hill interactive failure criterion.

    An extension of Hill's anisotropic yield criterion to brittle composite
    failure.  The failure index (FI) is:

        FI = (σ1/F1)² - (σ1·σ2)/F1² + (σ2/F2)² + (τ12/F12)²

    where F1 and F2 are selected as the tension or compression strength
    depending on the sign of the corresponding stress.

    Failure when FI ≥ 1.  Reserve factor:
        RF = 1 / √FI

    The cross-term -(σ1·σ2)/F1² introduces interaction between the fibre and
    transverse stress components.  Unlike Tsai-Wu, it does NOT distinguish
    tension vs compression through separate linear strength terms — the same
    F1 appears in both the quadratic and the cross term.

    Parameters
    ----------
    mat : PlyMaterial
    sig_12 : np.ndarray, shape (3,)
        Ply-axis stresses [σ1, σ2, τ12] [Pa].
    ply_index, angle_deg : for labelling.

    Returns
    -------
    PlyFailureResult
    """
    s1, s2, s12 = sig_12

    # Select t/c strength based on sign of each stress component
    F1 = mat.F1t if s1 >= 0 else mat.F1c
    F2 = mat.F2t if s2 >= 0 else mat.F2c

    # Tsai-Hill failure index
    FI = (s1/F1)**2 - (s1*s2)/F1**2 + (s2/F2)**2 + (s12/mat.F12)**2

    # Guard against FI ≤ 0 (zero stress state)
    rf_combined = 1.0 / np.sqrt(max(FI, 1e-30))

    # Individual margins for bookkeeping (not used in FI but useful to report)
    rf_1  = F1       / abs(s1)  if s1  != 0 else np.inf
    rf_2  = F2       / abs(s2)  if s2  != 0 else np.inf
    rf_12 = mat.F12  / abs(s12) if s12 != 0 else np.inf

    return PlyFailureResult(ply_index, angle_deg, "TsaiHill",
                            rf_1, rf_2, rf_12, rf_combined=rf_combined,
                            rf=rf_combined, failed=(rf_combined < 1.0))


# ---------------------------------------------------------------------------
# Tsai-Wu Criterion  (industry standard for composite sizing)
# ---------------------------------------------------------------------------

def tsai_wu(mat: PlyMaterial,
            sig_12: np.ndarray,
            ply_index: int = 0,
            angle_deg: float = 0.0,
            F12_star: Optional[float] = None) -> PlyFailureResult:
    """
    Tsai-Wu tensor polynomial failure criterion.

    The full second-order tensor polynomial in stress space:
        F1·σ1 + F2·σ2 + F11·σ1² + F22·σ2² + F66·τ12² + 2F12*·σ1·σ2 = 1

    Strength tensors:
        F1  = 1/F1t - 1/F1c         (linear fibre-direction term)
        F2  = 1/F2t - 1/F2c         (linear transverse term)
        F11 = 1/(F1t·F1c)           (quadratic fibre term)
        F22 = 1/(F2t·F2c)           (quadratic transverse term)
        F66 = 1/F12²                (quadratic shear term)
        F12*: interaction — requires biaxial test data; default uses Tsai approx.

    Reserve factor RF is the scalar multiplier on the load vector such that
    failure just occurs.  Substituting σ → RF·σ into the criterion gives a
    quadratic in RF:

        a·RF² + b·RF - 1 = 0

    where:
        a = F11·σ1² + F22·σ2² + F66·τ12² + 2F12*·σ1·σ2   (≥ 0 for physical cases)
        b = F1·σ1 + F2·σ2

    Positive root:
        RF = [-b + √(b² + 4a)] / (2a)

    Key advantages over Tsai-Hill:
      1. Accounts separately for different tensile/compressive strengths via
         the linear terms (F1, F2) — critical for CFRP where F1c << F1t
      2. Mathematically rigorous tensor formulation (invariant under coordinate
         rotation when all terms are included)
      3. Continuous RF — does not discontinuously switch allowable at σ = 0

    Parameters
    ----------
    mat : PlyMaterial
        Material with all five strength allowables defined.
    sig_12 : np.ndarray, shape (3,)
        Ply-axis stresses [σ1, σ2, τ12] [Pa].
    ply_index, angle_deg : for labelling.
    F12_star : float, optional
        Tsai-Wu interaction coefficient F12* [1/Pa²].
        Default: -0.5 / √(F1t·F1c·F2t·F2c)  (Tsai's conservative estimate).
        Set to 0 to disable interaction (reduces to decoupled criterion).

    Returns
    -------
    PlyFailureResult
    """
    s1, s2, s12 = sig_12

    # Strength tensor components
    F1  = 1.0/mat.F1t - 1.0/mat.F1c          # linear: distinguishes t/c in fibre dir
    F2  = 1.0/mat.F2t - 1.0/mat.F2c          # linear: distinguishes t/c in transverse
    F11 = 1.0/(mat.F1t * mat.F1c)            # quadratic fibre term
    F22 = 1.0/(mat.F2t * mat.F2c)            # quadratic transverse term
    F66 = 1.0/(mat.F12**2)                   # quadratic shear term

    # Interaction coefficient — experimental ideally, approximated here
    if F12_star is None:
        # Tsai's recommendation: -0.5 / geometric mean of all four strengths
        # Satisfies the stability requirement |F12*| ≤ √(F11·F22)
        F12_interaction = -0.5 / np.sqrt(mat.F1t * mat.F1c * mat.F2t * mat.F2c)
    else:
        F12_interaction = F12_star

    # Quadratic in RF: a·RF² + b·RF - 1 = 0
    a = F11*s1**2 + F22*s2**2 + F66*s12**2 + 2*F12_interaction*s1*s2
    b = F1*s1 + F2*s2

    discriminant = b**2 + 4*a
    if a == 0:
        # Degenerate case: only linear term present (very unlikely for real loads)
        rf_combined = (1.0 / b) if b > 0 else np.inf
    else:
        # Physical root (positive RF); discard the negative root
        rf_combined = (-b + np.sqrt(max(discriminant, 0))) / (2*a)

    # Individual directional margins (informational — help identify failure mode)
    rf_1  = mat.F1t / s1    if s1  > 0 else mat.F1c / abs(s1)  if s1  < 0 else np.inf
    rf_2  = mat.F2t / s2    if s2  > 0 else mat.F2c / abs(s2)  if s2  < 0 else np.inf
    rf_12 = mat.F12 / abs(s12) if s12 != 0 else np.inf

    return PlyFailureResult(ply_index, angle_deg, "TsaiWu",
                            rf_1, rf_2, rf_12, rf_combined=rf_combined,
                            rf=rf_combined, failed=(rf_combined < 1.0))


# ---------------------------------------------------------------------------
# Hashin (1980) Criterion  — physically motivated fiber/matrix separation
# ---------------------------------------------------------------------------

def hashin(mat: PlyMaterial,
           sig_12: np.ndarray,
           ply_index: int = 0,
           angle_deg: float = 0.0) -> PlyFailureResult:
    """
    Hashin (1980) failure criterion.

    Unlike Tsai-Wu (a single polynomial over all stresses), Hashin separates
    failure into four physically distinct modes:

      Fiber tension   (σ1 ≥ 0):  FI_ft = (σ1/F1t)²  + (τ12/F12)²
      Fiber compr.    (σ1 < 0):  FI_fc = (σ1/F1c)²
      Matrix tension  (σ2 ≥ 0):  FI_mt = (σ2/F2t)²  + (τ12/F12)²
      Matrix compr.   (σ2 < 0):  FI_mc = (σ2/F2c)²  + (τ12/F12)²

    Reserve factor for each mode: RF_i = 1 / sqrt(FI_i).
    Governing RF = min across active modes.

    The fiber mode and matrix mode RFs are reported separately in rf_1 and rf_2
    respectively, so the designer can see at a glance whether the fiber or the
    matrix is driving the design — a key advantage over single-polynomial criteria.

    Note on fiber compression: Hashin's original (1980) fiber compression mode
    does not include a shear term (unlike Hashin-Rotem). This is conservative for
    panels under combined in-plane compression and shear — if biaxial test data
    is available, the shear interaction term should be added.

    Parameters
    ----------
    mat : PlyMaterial
    sig_12 : np.ndarray, shape (3,)
        [σ1, σ2, τ12] in Pa.
    ply_index, angle_deg : for labelling.

    Returns
    -------
    PlyFailureResult
        rf_1  = governing fiber mode RF
        rf_2  = governing matrix mode RF
        rf_12 = shear-only reference RF  (F12 / |τ12|)
        rf    = min(rf_1, rf_2)

    Reference
    ---------
    Hashin, Z. (1980) Failure Criteria for Unidirectional Fiber Composites.
    Journal of Applied Mechanics, 47(2), 329–334.
    """
    s1, s2, s12 = sig_12
    eps = 1e-30   # guard against zero-stress singularities

    # ── fiber modes ──────────────────────────────────────────────────────────
    if s1 >= 0:
        FI_fiber = (s1 / mat.F1t)**2 + (s12 / mat.F12)**2
    else:
        FI_fiber = (s1 / mat.F1c)**2   # compression: no shear interaction (Hashin 1980)

    rf_1 = 1.0 / np.sqrt(max(FI_fiber, eps))

    # ── matrix modes ─────────────────────────────────────────────────────────
    if s2 >= 0:
        FI_matrix = (s2 / mat.F2t)**2 + (s12 / mat.F12)**2
    else:
        FI_matrix = (s2 / mat.F2c)**2 + (s12 / mat.F12)**2

    rf_2 = 1.0 / np.sqrt(max(FI_matrix, eps))

    # ── shear-only reference (informational) ─────────────────────────────────
    rf_12 = mat.F12 / abs(s12) if s12 != 0 else np.inf

    rf_combined = min(rf_1, rf_2)

    return PlyFailureResult(ply_index, angle_deg, "Hashin",
                            rf_1, rf_2, rf_12, rf_combined=rf_combined,
                            rf=rf_combined, failed=(rf_combined < 1.0))


# ---------------------------------------------------------------------------
# Laminate-level failure sweep
# ---------------------------------------------------------------------------

def check_laminate(laminate_response: dict,
                   plies,
                   criterion: str = "tsai_wu",
                   verbose: bool = True) -> list[PlyFailureResult]:
    """
    Evaluate a failure criterion across all plies in the laminate.

    Iterates over the ply stack and applies the selected criterion to each ply
    using the principal-axis stresses (or strains for max_strain) from the
    CLT response dict.

    The governing RF is the minimum across all plies — this is the first-ply
    failure (FPF) criterion.  A design is acceptable when governing RF ≥ 1.0,
    and typically targeted at RF ≥ 1.5 for metallic equivalent safety factors.

    Design interpretation of the output:
      - Ply with lowest RF → governs the design → needs to be addressed first
      - RF dominated by rf_2 (transverse) → consider adding 90° plies
      - RF dominated by rf_12 (shear) → consider adding ±45° plies
      - RF dominated by rf_1 (fibre compression) → check for buckling interaction

    Parameters
    ----------
    laminate_response : dict
        Output from Laminate.response().  Must contain 'ply_stress_12' and
        'ply_strain_12' lists.
    plies : list of Ply
        Must be the same list used to build the Laminate (matching order/length).
    criterion : str
        One of: 'tsai_wu' | 'tsai_hill' | 'max_stress' | 'max_strain'
    verbose : bool
        If True, print a formatted failure report to stdout.

    Returns
    -------
    list of PlyFailureResult
        One result per ply, in stack order (bottom to top).
    """
    # Dispatch table — makes it trivial to add new criteria in future
    _criteria = {
        'tsai_wu':    tsai_wu,
        'tsai_hill':  tsai_hill,
        'max_stress': max_stress,
        'max_strain': max_strain,
        'hashin':     hashin,
    }
    fn = _criteria[criterion.lower()]

    results = []
    for k, ply in enumerate(plies):
        mat = ply.material

        # Max stress / Tsai-Wu / Tsai-Hill / Hashin operate on stress in ply axes
        # Max strain operates on strain in ply axes
        if criterion in ('max_stress', 'tsai_hill', 'tsai_wu', 'hashin'):
            sig = laminate_response['ply_stress_12'][k]
            res = fn(mat, sig, ply_index=k, angle_deg=ply.angle_deg)
        else:  # max_strain
            eps = laminate_response['ply_strain_12'][k]
            res = fn(mat, eps, ply_index=k, angle_deg=ply.angle_deg)

        results.append(res)

    if verbose:
        print(f"\nFailure Analysis  –  {criterion}")
        print("-" * 65)
        for r in results:
            print(r)
        gov = min(results, key=lambda r: r.rf)
        print("-" * 65)
        print(f"  Governing RF = {gov.rf:.4f}  (ply {gov.ply_index}, "
              f"θ={gov.angle_deg}°)  →  {'FAIL' if gov.failed else 'OK'}")

    return results
