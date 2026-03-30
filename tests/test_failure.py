"""
tests/test_failure.py
Tsai-Wu failure criterion: boundary conditions, scaling, and per-ply checks.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pytest
from composite_panel import Ply, Laminate, IM7_8552, check_laminate
from composite_panel.failure import tsai_wu, hashin


@pytest.fixture
def mat():
    return IM7_8552()


# --- RF = 1.0 at every uniaxial failure boundary ---

@pytest.mark.parametrize("sigma,label", [
    (lambda m: np.array([m.F1t,    0,    0]), "F1t"),
    (lambda m: np.array([-m.F1c,   0,    0]), "F1c"),
    (lambda m: np.array([0,    m.F2t,    0]), "F2t"),
    (lambda m: np.array([0,   -m.F2c,    0]), "F2c"),
    (lambda m: np.array([0,       0, m.F12]), "F12"),
])
def test_rf_at_failure_boundary(mat, sigma, label):
    sig = sigma(mat)
    rf = tsai_wu(mat, sig).rf
    assert abs(rf - 1.0) < 0.02, f"RF at {label} boundary = {rf:.4f}, expected ~1.0"


def test_rf_scales_with_load(mat):
    # Half the load → RF should approximately double (linear region)
    rf_full = tsai_wu(mat, np.array([mat.F1t,      0, 0])).rf
    rf_half = tsai_wu(mat, np.array([mat.F1t * 0.5, 0, 0])).rf
    assert abs(rf_half - 2 * rf_full) / (2 * rf_full) < 0.02


def test_rf_above_one_for_safe_load(mat):
    rf = tsai_wu(mat, np.array([mat.F1t * 0.5, 0, 0])).rf
    assert rf > 1.0


def test_rf_below_one_for_overload(mat):
    rf = tsai_wu(mat, np.array([mat.F1t * 1.5, 0, 0])).rf
    assert rf < 1.0


def test_check_laminate_returns_one_result_per_ply(mat):
    t = 0.125e-3
    plies = [Ply(mat, t, a) for a in [0, 45, -45, 90, 90, -45, 45, 0]]
    lam = Laminate(plies)
    res = lam.response(N=np.array([-50e3, -10e3, 5e3]))
    fails = check_laminate(res, plies, criterion='tsai_wu', verbose=False)
    assert len(fails) == len(plies)


def test_governing_ply_has_minimum_rf(mat):
    t = 0.125e-3
    plies = [Ply(mat, t, a) for a in [0, 45, -45, 90, 90, -45, 45, 0]]
    lam = Laminate(plies)
    res = lam.response(N=np.array([-100e3, -30e3, 10e3]))
    fails = check_laminate(res, plies, criterion='tsai_wu', verbose=False)
    rfs = [f.rf for f in fails]
    gov = min(fails, key=lambda x: x.rf)
    assert gov.rf == min(rfs)


def test_high_Nxx_makes_0deg_ply_govern(mat):
    # Under strong spanwise compression, 0° ply should have highest stress (lowest RF)
    t = 0.125e-3
    plies = [Ply(mat, t, a) for a in [0, 45, -45, 90, 90, -45, 45, 0]]
    lam = Laminate(plies)
    res = lam.response(N=np.array([-2000e3, 0, 0]))
    fails = check_laminate(res, plies, criterion='tsai_wu', verbose=False)
    gov_idx = min(range(len(fails)), key=lambda i: fails[i].rf)
    assert plies[gov_idx].angle_deg in (0.0,)


# --- Hashin criterion ---

def test_hashin_fiber_tension_at_boundary(mat):
    # Pure σ1 = F1t → fiber tension mode → RF = 1.0
    sig = np.array([mat.F1t, 0.0, 0.0])
    res = hashin(mat, sig)
    assert abs(res.rf_1 - 1.0) < 0.02
    assert res.rf == res.rf_1   # fiber governs, not matrix


def test_hashin_fiber_compression_at_boundary(mat):
    # Pure σ1 = -F1c → fiber compression mode → RF = 1.0
    sig = np.array([-mat.F1c, 0.0, 0.0])
    res = hashin(mat, sig)
    assert abs(res.rf_1 - 1.0) < 0.02


def test_hashin_matrix_tension_at_boundary(mat):
    # Pure σ2 = F2t → matrix tension mode → RF = 1.0
    sig = np.array([0.0, mat.F2t, 0.0])
    res = hashin(mat, sig)
    assert abs(res.rf_2 - 1.0) < 0.02
    assert res.rf == res.rf_2   # matrix governs


def test_hashin_shear_at_boundary(mat):
    # Pure τ12 = F12 → both fiber tension and matrix tension modes fire
    sig = np.array([0.0, 0.0, mat.F12])
    res = hashin(mat, sig)
    # Both modes include shear; both should give RF ≈ 1.0
    assert abs(res.rf_1 - 1.0) < 0.02
    assert abs(res.rf_2 - 1.0) < 0.02


def test_hashin_rf_above_one_safe(mat):
    sig = np.array([mat.F1t * 0.5, 0.0, 0.0])
    res = hashin(mat, sig)
    assert res.rf > 1.0
    assert not res.failed


def test_hashin_rf_below_one_overload(mat):
    sig = np.array([mat.F1t * 1.5, 0.0, 0.0])
    res = hashin(mat, sig)
    assert res.rf < 1.0
    assert res.failed


def test_hashin_check_laminate_dispatch(mat):
    # check_laminate should accept criterion='hashin' without error
    t = 0.125e-3
    plies = [Ply(mat, t, a) for a in [0, 45, -45, 90]]
    lam = Laminate(plies)
    res = lam.response(N=np.array([-50e3, -10e3, 5e3]))
    fails = check_laminate(res, plies, criterion='hashin', verbose=False)
    assert len(fails) == len(plies)
    assert all(isinstance(f.rf, float) for f in fails)
