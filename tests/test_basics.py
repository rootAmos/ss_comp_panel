"""
tests/test_basics.py
Basic sanity checks for composite_panel.
Run with:  python -m pytest tests/ -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from composite_panel import Ply, Laminate, IM7_8552, T300_5208
from composite_panel import check_laminate, tsai_wu
from composite_panel import supersonic_panel_loads


def test_ply_Q_symmetric():
    mat = IM7_8552()
    ply = Ply(mat, 0.125e-3, 0.0)
    Q = ply.material.Q
    assert np.allclose(Q, Q.T), "Q matrix not symmetric"


def test_Qbar_0_deg_equals_Q():
    mat = IM7_8552()
    ply = Ply(mat, 0.125e-3, 0.0)
    assert np.allclose(ply.Q_bar, ply.material.Q, rtol=1e-10)


def test_Qbar_90_deg():
    mat = IM7_8552()
    p0  = Ply(mat, 0.125e-3,  0.0)
    p90 = Ply(mat, 0.125e-3, 90.0)
    # Q11 at 0° should equal Q22 at 90° (transverse becomes fibre direction)
    assert abs(p0.Q_bar[0, 0] - p90.Q_bar[1, 1]) < 1.0


def test_symmetric_laminate_zero_B():
    mat   = IM7_8552()
    t     = 0.125e-3
    plies = [Ply(mat, t, a) for a in [-45, 0, 45, 90, 90, 45, 0, -45]]
    lam   = Laminate(plies)
    assert np.allclose(lam.B, 0, atol=1e-3), "B matrix should be ~zero for symmetric laminate"


def test_ABD_invertible():
    mat   = T300_5208()
    t     = 0.125e-3
    plies = [Ply(mat, t, a) for a in [0, 90, 0, 90]]
    lam   = Laminate(plies)
    det   = np.linalg.det(lam.ABD)
    assert abs(det) > 1e-10, "ABD matrix is singular"


def test_response_zero_loads():
    mat   = IM7_8552()
    plies = [Ply(mat, 0.125e-3, a) for a in [0, 45, -45, 90] * 2]
    lam   = Laminate(plies)
    res   = lam.response()
    assert np.allclose(res['eps0'],  0)
    assert np.allclose(res['kappa'], 0)


def test_tsai_wu_high_load_fails():
    """Apply a load 100× the F1t strength – should definitely fail."""
    mat   = IM7_8552()
    plies = [Ply(mat, 0.125e-3, 0.0)]
    lam   = Laminate(plies)
    N_extreme = np.array([mat.F1t * 100 * plies[0].thickness, 0, 0])
    res   = lam.response(N=N_extreme)
    results = check_laminate(res, plies, criterion='tsai_wu', verbose=False)
    assert results[0].failed, "Expected failure under extreme load"


def test_tsai_wu_low_load_passes():
    """Apply a small load – should pass easily."""
    mat   = IM7_8552()
    plies = [Ply(mat, 0.125e-3, a) for a in [-45, 0, 45, 90, 90, 45, 0, -45]]
    lam   = Laminate(plies)
    N_small = np.array([1e3, 0, 0])   # 1 kN/m – trivially safe
    res   = lam.response(N=N_small)
    results = check_laminate(res, plies, criterion='tsai_wu', verbose=False)
    assert not any(r.failed for r in results)


def test_supersonic_loads_physical():
    loads = supersonic_panel_loads(
        mach=1.6, altitude_m=15000, alpha_deg=3.0,
        panel_chord=0.8, panel_span=0.5
    )
    # Compression on upper skin → negative Nyy
    assert loads.Nyy < 0, "Expected compressive Nyy on upper surface"
    assert abs(loads.Nxx) > 0
    assert abs(loads.Nxy) > 0


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
