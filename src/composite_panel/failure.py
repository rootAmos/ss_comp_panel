"""
composite_panel.failure
-----------------------
Ply-level failure criteria.

Implements:
  - Maximum Stress
  - Maximum Strain
  - Tsai-Hill
  - Tsai-Wu  (most commonly used in industry)

All criteria work on principal ply axes stresses/strains.
Returns a Failure Reserve Factor (RF): RF < 1.0 → failure.

Reference:
    Kassapoglou, C. – Design and Analysis of Composite Structures
    (Wiley, 2013), Ch. 5
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional
from .ply import PlyMaterial


@dataclass
class PlyFailureResult:
    """Failure assessment for a single ply."""
    ply_index:   int
    angle_deg:   float
    criterion:   str

    # Individual margin components (positive = safe)
    rf_1:  float   # fibre direction
    rf_2:  float   # transverse direction
    rf_12: float   # shear
    rf_combined: float   # combined index (Tsai-Hill / Tsai-Wu)

    # Governing reserve factor
    rf: float
    failed: bool

    def __str__(self) -> str:
        status = "FAIL" if self.failed else "OK  "
        return (f"  [{status}] Ply {self.ply_index:2d}  θ={self.angle_deg:5.1f}°  "
                f"RF={self.rf:.4f}  (1:{self.rf_1:.3f}  2:{self.rf_2:.3f}  "
                f"12:{self.rf_12:.3f}  comb:{self.rf_combined:.4f})")


# ---------------------------------------------------------------------------
# Maximum Stress
# ---------------------------------------------------------------------------

def max_stress(mat: PlyMaterial,
               sig_12: np.ndarray,
               ply_index: int = 0,
               angle_deg: float = 0.0) -> PlyFailureResult:
    """
    Maximum stress failure criterion.

    sig_12 : (σ1, σ2, τ12) in principal axes [Pa]
    Returns PlyFailureResult with RF = min(individual RFs).
    """
    s1, s2, s12 = sig_12

    rf_1  = mat.F1t / s1    if s1  > 0 else mat.F1c / abs(s1)  if s1  < 0 else np.inf
    rf_2  = mat.F2t / s2    if s2  > 0 else mat.F2c / abs(s2)  if s2  < 0 else np.inf
    rf_12 = mat.F12 / abs(s12) if s12 != 0 else np.inf

    rf = min(rf_1, rf_2, rf_12)
    return PlyFailureResult(ply_index, angle_deg, "MaxStress",
                            rf_1, rf_2, rf_12, rf_combined=rf,
                            rf=rf, failed=(rf < 1.0))


# ---------------------------------------------------------------------------
# Maximum Strain
# ---------------------------------------------------------------------------

def max_strain(mat: PlyMaterial,
               eps_12: np.ndarray,
               ply_index: int = 0,
               angle_deg: float = 0.0) -> PlyFailureResult:
    """
    Maximum strain failure criterion.
    eps_12 : (ε1, ε2, γ12) in principal axes [-]
    """
    e1, e2, g12 = eps_12

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
# Tsai-Hill
# ---------------------------------------------------------------------------

def tsai_hill(mat: PlyMaterial,
              sig_12: np.ndarray,
              ply_index: int = 0,
              angle_deg: float = 0.0) -> PlyFailureResult:
    """
    Tsai-Hill failure criterion.
    RF = 1 / sqrt(failure_index).
    """
    s1, s2, s12 = sig_12

    F1 = mat.F1t if s1 >= 0 else mat.F1c
    F2 = mat.F2t if s2 >= 0 else mat.F2c

    FI = (s1/F1)**2 - (s1*s2)/F1**2 + (s2/F2)**2 + (s12/mat.F12)**2

    rf_combined = 1.0 / np.sqrt(max(FI, 1e-30))

    # Individual margins for bookkeeping
    rf_1  = F1  / abs(s1)  if s1  != 0 else np.inf
    rf_2  = F2  / abs(s2)  if s2  != 0 else np.inf
    rf_12 = mat.F12 / abs(s12) if s12 != 0 else np.inf

    return PlyFailureResult(ply_index, angle_deg, "TsaiHill",
                            rf_1, rf_2, rf_12, rf_combined=rf_combined,
                            rf=rf_combined, failed=(rf_combined < 1.0))


# ---------------------------------------------------------------------------
# Tsai-Wu  (industry standard)
# ---------------------------------------------------------------------------

def tsai_wu(mat: PlyMaterial,
            sig_12: np.ndarray,
            ply_index: int = 0,
            angle_deg: float = 0.0,
            F12_star: Optional[float] = None) -> PlyFailureResult:
    """
    Tsai-Wu tensor polynomial failure criterion.

    F12_star : interaction coefficient (default = -0.5 / sqrt(F1t*F1c*F2t*F2c))
               Set to 0 to use Tsai-Hill-equivalent interaction.

    RF is found by solving the quadratic:
        FI_quadratic * RF² + FI_linear * RF - 1 = 0
    → RF = [ -FI_linear + sqrt(FI_linear² + 4*FI_quadratic) ] / (2*FI_quadratic)
    """
    s1, s2, s12 = sig_12

    F1  = 1.0/mat.F1t - 1.0/mat.F1c
    F2  = 1.0/mat.F2t - 1.0/mat.F2c
    F11 = 1.0/(mat.F1t * mat.F1c)
    F22 = 1.0/(mat.F2t * mat.F2c)
    F66 = 1.0/(mat.F12**2)

    if F12_star is None:
        F12_interaction = -0.5 / np.sqrt(mat.F1t * mat.F1c * mat.F2t * mat.F2c)
    else:
        F12_interaction = F12_star

    # Quadratic: a*RF² + b*RF - 1 = 0
    a = F11*s1**2 + F22*s2**2 + F66*s12**2 + 2*F12_interaction*s1*s2
    b = F1*s1 + F2*s2

    discriminant = b**2 + 4*a
    if a == 0:
        rf_combined = (1.0 / b) if b > 0 else np.inf
    else:
        rf_combined = (-b + np.sqrt(max(discriminant, 0))) / (2*a)

    rf_1  = mat.F1t / s1    if s1  > 0 else mat.F1c / abs(s1)  if s1  < 0 else np.inf
    rf_2  = mat.F2t / s2    if s2  > 0 else mat.F2c / abs(s2)  if s2  < 0 else np.inf
    rf_12 = mat.F12 / abs(s12) if s12 != 0 else np.inf

    return PlyFailureResult(ply_index, angle_deg, "TsaiWu",
                            rf_1, rf_2, rf_12, rf_combined=rf_combined,
                            rf=rf_combined, failed=(rf_combined < 1.0))


# ---------------------------------------------------------------------------
# Laminate-level sweep
# ---------------------------------------------------------------------------

def check_laminate(laminate_response: dict,
                   plies,
                   criterion: str = "tsai_wu",
                   verbose: bool = True) -> list[PlyFailureResult]:
    """
    Run failure analysis across all plies.

    Parameters
    ----------
    laminate_response : dict returned by Laminate.response()
    plies             : list of Ply objects
    criterion         : 'tsai_wu' | 'tsai_hill' | 'max_stress' | 'max_strain'
    verbose           : print results to stdout

    Returns
    -------
    list of PlyFailureResult
    """
    _criteria = {
        'tsai_wu':    tsai_wu,
        'tsai_hill':  tsai_hill,
        'max_stress': max_stress,
        'max_strain': max_strain,
    }
    fn = _criteria[criterion.lower()]

    results = []
    for k, ply in enumerate(plies):
        mat = ply.material
        if criterion in ('max_stress', 'tsai_hill', 'tsai_wu'):
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
