"""
Ply-level failure criteria: max stress, max strain, Tsai-Hill, Tsai-Wu, Hashin.

All criteria operate at first-ply failure (FPF) on principal ply-axis stresses.
RF = allowable / applied; RF < 1 means failed.

Ref: Tsai & Wu (1971), Hashin (1980), Kassapoglou (2013) Ch. 5
"""

from __future__ import annotations
import aerosandbox.numpy as np
from dataclasses import dataclass
from typing import Optional
from composite_panel.ply import PlyMaterial


@dataclass
class PlyFailureResult:
    """Failure result for one ply: directional RFs + governing combined RF."""
    ply_index:   int
    angle_deg:   float
    criterion:   str
    rf_1:  float
    rf_2:  float
    rf_12: float
    rf_combined: float
    rf: float
    failed: bool

    def __str__(self) -> str:
        status = "FAIL" if self.failed else "OK  "
        return (f"  [{status}] Ply {self.ply_index:2d}  theta={self.angle_deg:5.1f}deg  "
                f"RF={self.rf:.4f}  (1:{self.rf_1:.3f}  2:{self.rf_2:.3f}  "
                f"12:{self.rf_12:.3f}  comb:{self.rf_combined:.4f})")


def max_stress(mat: PlyMaterial, sig_12: np.ndarray,
               ply_index: int = 0, angle_deg: float = 0.0) -> PlyFailureResult:
    """Maximum stress criterion.  Jones (1999), Sec. 5.2."""
    s1, s2, s12 = sig_12
    rf_1  = mat.F1t / s1       if s1  > 0 else mat.F1c / abs(s1)  if s1  < 0 else np.inf
    rf_2  = mat.F2t / s2       if s2  > 0 else mat.F2c / abs(s2)  if s2  < 0 else np.inf
    rf_12 = mat.F12 / abs(s12) if s12 != 0 else np.inf
    rf = min(rf_1, rf_2, rf_12)
    return PlyFailureResult(ply_index, angle_deg, "MaxStress",
                            rf_1, rf_2, rf_12, rf_combined=rf,
                            rf=rf, failed=(rf < 1.0))


def max_strain(mat: PlyMaterial, eps_12: np.ndarray,
               ply_index: int = 0, angle_deg: float = 0.0) -> PlyFailureResult:
    """Maximum strain criterion.  Jones (1999), Sec. 5.3."""
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


def tsai_hill(mat: PlyMaterial, sig_12: np.ndarray,
              ply_index: int = 0, angle_deg: float = 0.0) -> PlyFailureResult:
    """Azzi-Tsai-Hill criterion.  Azzi & Tsai (1965); Jones (1999), Eq. 5.12."""
    s1, s2, s12 = sig_12
    F1 = mat.F1t if s1 >= 0 else mat.F1c
    F2 = mat.F2t if s2 >= 0 else mat.F2c
    FI = (s1/F1)**2 - (s1*s2)/F1**2 + (s2/F2)**2 + (s12/mat.F12)**2
    rf_combined = 1.0 / np.sqrt(max(FI, 1e-30))
    rf_1  = F1       / abs(s1)  if s1  != 0 else np.inf
    rf_2  = F2       / abs(s2)  if s2  != 0 else np.inf
    rf_12 = mat.F12  / abs(s12) if s12 != 0 else np.inf
    return PlyFailureResult(ply_index, angle_deg, "TsaiHill",
                            rf_1, rf_2, rf_12, rf_combined=rf_combined,
                            rf=rf_combined, failed=(rf_combined < 1.0))


def tsai_wu(mat: PlyMaterial, sig_12: np.ndarray,
            ply_index: int = 0, angle_deg: float = 0.0,
            F12_star: Optional[float] = None) -> PlyFailureResult:
    """Tsai-Wu tensor polynomial.  Tsai & Wu (1971), J. Composite Materials; Jones (1999), Eq. 5.20.
    F12* defaults to -0.5/sqrt(F1t*F1c*F2t*F2c) per Tsai & Hahn (1980)."""
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


def hashin(mat: PlyMaterial, sig_12: np.ndarray,
           ply_index: int = 0, angle_deg: float = 0.0) -> PlyFailureResult:
    """Hashin (1980), J. Applied Mechanics, Eqs. 1-4.
    Fiber compression: no shear term (original Hashin, not Hashin-Rotem variant)."""
    s1, s2, s12 = sig_12
    eps = 1e-30

    if s1 >= 0:
        FI_fiber = (s1 / mat.F1t)**2 + (s12 / mat.F12)**2
    else:
        FI_fiber = (s1 / mat.F1c)**2
    rf_1 = 1.0 / np.sqrt(max(FI_fiber, eps))

    if s2 >= 0:
        FI_matrix = (s2 / mat.F2t)**2 + (s12 / mat.F12)**2
    else:
        FI_matrix = (s2 / mat.F2c)**2 + (s12 / mat.F12)**2
    rf_2 = 1.0 / np.sqrt(max(FI_matrix, eps))

    rf_12 = mat.F12 / abs(s12) if s12 != 0 else np.inf
    rf_combined = min(rf_1, rf_2)

    return PlyFailureResult(ply_index, angle_deg, "Hashin",
                            rf_1, rf_2, rf_12, rf_combined=rf_combined,
                            rf=rf_combined, failed=(rf_combined < 1.0))


def check_laminate(laminate_response: dict, plies,
                   criterion: str = "tsai_wu",
                   verbose: bool = True) -> list[PlyFailureResult]:
    """Evaluate failure criterion across all plies.  Returns list of PlyFailureResult."""
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
        if criterion in ('max_stress', 'tsai_hill', 'tsai_wu', 'hashin'):
            sig = laminate_response['ply_stress_12'][k]
            res = fn(mat, sig, ply_index=k, angle_deg=ply.angle_deg)
        else:
            eps = laminate_response['ply_strain_12'][k]
            res = fn(mat, eps, ply_index=k, angle_deg=ply.angle_deg)
        results.append(res)

    if verbose:
        print(f"\nFailure Analysis  -  {criterion}")
        print("-" * 65)
        for r in results:
            print(r)
        gov = min(results, key=lambda r: r.rf)
        print("-" * 65)
        print(f"  Governing RF = {gov.rf:.4f}  (ply {gov.ply_index}, "
              f"theta={gov.angle_deg}deg)  ->  {'FAIL' if gov.failed else 'OK'}")

    return results


if __name__ == "__main__":
    import sys as _sys, os as _os
    _sys.stdout.reconfigure(encoding="utf-8")
    _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
    from composite_panel.ply import Ply, IM7_8552
    from composite_panel.laminate import Laminate

    mat   = IM7_8552()
    t_ply = 0.125e-3
    angles = [0, 45, -45, 90, 90, -45, 45, 0]
    plies  = [Ply(mat, t_ply, a) for a in angles]
    lam    = Laminate(plies)

    N = np.array([-280e3, -115e3, 42e3])
    M = np.array([   60.0,   0.0,  0.0])
    res = lam.response(N=N, M=M)

    print(f"Applied N = [{N[0]/1e3:.0f}, {N[1]/1e3:.0f}, {N[2]/1e3:.0f}] kN/m")
    for criterion in ["tsai_wu", "tsai_hill", "max_stress", "hashin"]:
        check_laminate(res, plies, criterion=criterion)
