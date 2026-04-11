"""
tests/test_clt.py
CLT: ABD matrix construction and laminate response against known analytical results.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pytest
from composite_panel import Ply, Laminate, IM7_8552


@pytest.fixture
def mat():
    return IM7_8552()


@pytest.fixture
def t():
    return 0.125e-3


def test_A11_unidirectional(mat, t):
    # [0]8: A11 = Q11 * total_thickness  (exact, no approximation)
    n = 8
    lam = Laminate([Ply(mat, t, 0.0)] * n)
    assert abs(lam.A[0, 0] - mat.Q[0, 0] * n * t) / (mat.Q[0, 0] * n * t) < 1e-10


def test_90deg_rotation_symmetry(mat, t):
    # [90]8 A22 should equal [0]8 A11  --  same material, just rotated
    lam_0  = Laminate([Ply(mat, t, 0.0)]  * 8)
    lam_90 = Laminate([Ply(mat, t, 90.0)] * 8)
    assert abs(lam_90.A[1, 1] - lam_0.A[0, 0]) / lam_0.A[0, 0] < 1e-6


def test_cross_ply_biaxial_symmetry(mat, t):
    # [0/90]s: A11 == A22
    lam = Laminate([Ply(mat, t, a) for a in [0, 90, 90, 0]])
    assert abs(lam.A[0, 0] - lam.A[1, 1]) / lam.A[0, 0] < 1e-6


def test_balanced_A16_A26_zero(mat, t):
    # [+45/-45]s: A16 and A26 must be exactly zero (balance condition)
    lam = Laminate([Ply(mat, t, a) for a in [45, -45, -45, 45]])
    assert abs(lam.A[0, 2]) < 1.0   # [N/m]
    assert abs(lam.A[1, 2]) < 1.0


def test_symmetric_B_zero(mat, t):
    # Any symmetric layup: B = 0 identically
    lam = Laminate([Ply(mat, t, a) for a in [-45, 0, 45, 90, 90, 45, 0, -45]])
    assert np.abs(lam.B).max() < 1e-3   # [N]  --  numerical zero


def test_D_cubic_thickness_scaling(mat):
    # D ~ h^3: doubling ply thickness -> 8x D
    lam1 = Laminate([Ply(mat, 0.125e-3, 0.0)] * 8)
    lam2 = Laminate([Ply(mat, 0.250e-3, 0.0)] * 8)
    ratio = lam2.D[0, 0] / lam1.D[0, 0]
    assert abs(ratio - 8.0) / 8.0 < 0.01


def test_quasi_isotropic_A_matrix(mat, t):
    # [0/+45/-45/90]s: A11 == A22 and A16 ~= 0 (quasi-isotropic condition)
    lam = Laminate([Ply(mat, t, a) for a in [0, 45, -45, 90, 90, -45, 45, 0]])
    assert abs(lam.A[0, 0] - lam.A[1, 1]) / lam.A[0, 0] < 1e-6
    assert abs(lam.A[0, 2]) < 1.0


def test_zero_load_zero_strain(mat, t):
    lam = Laminate([Ply(mat, t, a) for a in [0, 45, -45, 90, 90, -45, 45, 0]])
    res = lam.response()
    assert np.allclose(res['eps0'],  0, atol=1e-20)
    assert np.allclose(res['kappa'], 0, atol=1e-20)


def test_uniaxial_Nxx_gives_nonzero_strain(mat, t):
    lam = Laminate([Ply(mat, t, a) for a in [0, 45, -45, 90, 90, -45, 45, 0]])
    res = lam.response(N=np.array([-100e3, 0.0, 0.0]))
    # Nxx < 0 -> compressive strain in x -> eps0[0] < 0
    assert res['eps0'][0] < 0


def test_ABD_nonsingular(mat, t):
    lam = Laminate([Ply(mat, t, a) for a in [0, 45, -45, 90, 90, -45, 45, 0]])
    assert abs(np.linalg.det(lam.ABD)) > 1e-10


def test_cross_ply_A11_exact(mat, t):
    # [0/90]s repeating: A11 = (Q11 + Q22)/2 * h  (exact for 50/50 layup)
    plies = [Ply(mat, t, a) for a in [0, 90, 0, 90, 90, 0, 90, 0]]
    lam = Laminate(plies)
    ref = 0.5 * (mat.Q[0, 0] + mat.Q[1, 1]) * lam.thickness
    assert abs(lam.A[0, 0] - ref) / ref < 1e-6
