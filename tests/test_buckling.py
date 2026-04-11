"""
tests/test_buckling.py
Buckling: closed-form benchmark vs Timoshenko, monotonicity checks.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import math
import warnings
import numpy as np
import pytest
from composite_panel import Ply, Laminate, IM7_8552
from composite_panel.buckling import Nxx_cr, buckling_rf


@pytest.fixture
def mat():
    return IM7_8552()


@pytest.fixture
def panel():
    return dict(a=0.50, b=0.20)   # rib pitch x stringer pitch [m]


def _lam(mat, t, angle, n=8):
    return Laminate([Ply(mat, t, angle)] * n)


def test_Nxx_cr_exact_unidirectional(mat, panel):
    # [0]8: compare against Timoshenko simply-supported formula directly
    t = 0.125e-3
    n = 8
    lam = _lam(mat, t, 0.0, n)
    h = lam.thickness
    a, b = panel['a'], panel['b']

    D11 = mat.Q[0, 0] * h**3 / 12
    D12 = mat.Q[0, 1] * h**3 / 12
    D22 = mat.Q[1, 1] * h**3 / 12
    D66 = mat.Q[2, 2] * h**3 / 12

    Ncr_exact = min(
        (math.pi**2 / b**2) * (D11 * (m * b / a)**2 + 2 * (D12 + 2 * D66)
                                 + D22 * (a / (m * b))**2)
        for m in range(1, 9)
    )
    Ncr_model = Nxx_cr(lam.D, a, b)
    assert abs(Ncr_model - Ncr_exact) / Ncr_exact < 1e-4


def test_Nxx_cr_cubic_thickness_scaling(mat, panel):
    # D ~ h^3 -> Nxx_cr ~ h^3: doubling thickness -> 8x critical load
    a, b = panel['a'], panel['b']
    lam1 = _lam(mat, 0.125e-3, 0.0)
    lam2 = _lam(mat, 0.250e-3, 0.0)
    ratio = Nxx_cr(lam2.D, a, b) / Nxx_cr(lam1.D, a, b)
    assert abs(ratio - 8.0) / 8.0 < 0.01


def test_Nxx_cr_narrower_panel_higher(mat, panel):
    # Narrower panel (smaller b) -> higher Nxx_cr (b^2 in denominator)
    a = panel['a']
    lam = _lam(mat, 0.125e-3, 0.0)
    Ncr_wide   = Nxx_cr(lam.D, a, panel['b'])
    Ncr_narrow = Nxx_cr(lam.D, a, panel['b'] / 2)
    assert Ncr_narrow > Ncr_wide


def test_buckling_rf_above_one_light_load(mat, panel):
    a, b = panel['a'], panel['b']
    lam = _lam(mat, 0.125e-3, 0.0)
    N = np.array([-1e3, 0.0, 0.0])    # well below Nxx_cr (~2.3 kN/m)  --  should be safe
    rf = buckling_rf(N, lam.D, a, b)
    assert rf > 1.0


def test_buckling_rf_below_one_heavy_load(mat, panel):
    a, b = panel['a'], panel['b']
    lam = _lam(mat, 0.125e-3, 0.0)   # very thin
    N = np.array([-1e9, 0.0, 0.0])    # extreme load  --  should buckle
    rf = buckling_rf(N, lam.D, a, b)
    assert rf < 1.0


def test_buckling_rf_tension_not_critical(mat, panel):
    # Tensile Nxx should not cause buckling
    a, b = panel['a'], panel['b']
    lam = _lam(mat, 0.125e-3, 0.0)
    N_tension = np.array([500e3, 0.0, 0.0])
    rf = buckling_rf(N_tension, lam.D, a, b)
    assert rf > 10.0   # large RF  --  tension never governs buckling


def test_rr_knockdown_unbalanced(mat, panel):
    # Unbalanced laminate (D16, D26 != 0) should have lower Ncr than orthotropic formula
    a, b = panel['a'], panel['b']
    lam = Laminate([Ply(mat, 0.125e-3, 30.0)])   # unbalanced: no -30deg partner
    Ncr_ortho = Nxx_cr(lam.D, a, b)              # orthotropic (ignores D16/D26)
    from composite_panel.buckling import _rr_Nxx_cr
    Ncr_rr = _rr_Nxx_cr(lam.D, a, b)             # Rayleigh-Ritz (includes D16/D26)
    assert Ncr_rr < Ncr_ortho   # coupling always reduces Ncr


def test_rr_matches_orthotropic_cross_ply(mat, panel):
    # Cross-ply [0/90/90/0] -> D16 = D26 = 0, RR should match orthotropic exactly
    a, b = panel['a'], panel['b']
    plies = [Ply(mat, 0.125e-3, ang) for ang in [0, 90, 90, 0]]
    lam = Laminate(plies)
    Ncr_ortho = Nxx_cr(lam.D, a, b)
    from composite_panel.buckling import _rr_Nxx_cr
    Ncr_rr = _rr_Nxx_cr(lam.D, a, b)
    assert abs(Ncr_rr - Ncr_ortho) / Ncr_ortho < 0.01  # <1% difference


def test_buckling_rf_warns_on_bend_twist(mat, panel):
    # Significant D16/D26 should emit a warning from the exact buckling RF path.
    a, b = panel['a'], panel['b']
    lam = Laminate([Ply(mat, 0.125e-3, 30.0)])
    N = np.array([-1e3, 0.0, 0.0])
    with pytest.warns(UserWarning, match="bend-twist coupling"):
        buckling_rf(N, lam.D, a, b)
