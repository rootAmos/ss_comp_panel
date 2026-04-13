"""
tests/test_aero_loads.py
Pressure model routing and regime boundaries.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import math
import numpy as np

from composite_panel.aero_loads import (
    panel_pressure,
    prandtl_glauert_panel_pressure,
    ackeret_panel_pressure,
    hypersonic_panel_pressure,
)


def test_panel_pressure_subsonic_matches_prandtl_glauert():
    mach = 0.6
    alpha_deg = 3.0
    q_inf = 42_000.0
    ref = prandtl_glauert_panel_pressure(mach, alpha_deg, q_inf)
    got = panel_pressure(mach, alpha_deg, q_inf)
    assert math.isclose(got, ref, rel_tol=1e-12)


def test_panel_pressure_supersonic_matches_ackeret():
    mach = 1.7
    alpha_deg = 3.0
    q_inf = 52_000.0
    ref = ackeret_panel_pressure(mach, alpha_deg, q_inf)
    got = panel_pressure(mach, alpha_deg, q_inf)
    assert math.isclose(got, ref, rel_tol=1e-12)


def test_panel_pressure_hypersonic_matches_modified_newtonian():
    mach = 12.0     # well above the Mach-5 handoff, pure Newtonian
    alpha_deg = 3.0
    q_inf = 88_000.0
    ref = hypersonic_panel_pressure(mach, alpha_deg, q_inf)
    got = panel_pressure(mach, alpha_deg, q_inf)
    assert math.isclose(got, ref, rel_tol=1e-12)


def test_panel_pressure_hypersonic_switch_occurs_above_mach_five():
    mach = 5.1
    alpha_deg = 3.0
    q_inf = 88_000.0
    ref = hypersonic_panel_pressure(mach, alpha_deg, q_inf)
    got = panel_pressure(mach, alpha_deg, q_inf)
    assert math.isclose(got, ref, rel_tol=1e-12)


def test_panel_pressure_transonic_gap_raises():
    with np.testing.assert_raises(ValueError):
        panel_pressure(1.0, 3.0, 50_000.0)
