"""
tests/test_cp_integration.py
Cp integration pipeline tests, including real ONERA M6 data.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pytest

from composite_panel.cp_integration import (
    CpSection, CpField, section_coefficients,
    integrate_cp_field, load_cp_csv, load_cp_tecplot,
)

SCRIPTS = os.path.join(os.path.dirname(__file__), '..', 'scripts')


# == CpSection =============================================================

class TestCpSection:
    def test_from_delta_cp(self):
        sec = CpSection(
            x_over_c=np.linspace(0, 1, 20),
            delta_Cp=np.ones(20) * 0.5,
            eta=0.4,
        )
        assert sec.eta == 0.4
        assert len(sec.x_over_c) == 20

    def test_from_surfaces(self):
        x = np.linspace(0, 1, 20)
        Cp_upper = -0.5 * np.ones(20)
        Cp_lower = 0.1 * np.ones(20)
        sec = CpSection.from_surfaces(x, Cp_upper, x, Cp_lower, eta=0.3)
        np.testing.assert_allclose(sec.delta_Cp, 0.6, atol=1e-12)

    def test_from_surfaces_different_grids(self):
        """Upper and lower on different x grids still works."""
        x_u = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        x_l = np.array([0.0, 0.3, 0.6, 1.0])
        Cp_u = -0.4 * np.ones(5)
        Cp_l = 0.1 * np.ones(4)
        sec = CpSection.from_surfaces(x_u, Cp_u, x_l, Cp_l, eta=0.5)
        assert len(sec.x_over_c) >= 4
        assert np.all(sec.delta_Cp > 0)


# == section_coefficients ===================================================

class TestSectionCoefficients:
    def test_uniform_delta_cp(self):
        """Uniform delta_Cp of 1.0 => Cn = 1.0, Cm_le = -0.5."""
        sec = CpSection(
            x_over_c=np.linspace(0, 1, 200),
            delta_Cp=np.ones(200),
            eta=0.5,
        )
        Cn, Cm_le = section_coefficients(sec)
        assert Cn == pytest.approx(1.0, abs=0.01)
        assert Cm_le == pytest.approx(-0.5, abs=0.01)

    def test_zero_cp(self):
        sec = CpSection(
            x_over_c=np.linspace(0, 1, 20),
            delta_Cp=np.zeros(20),
            eta=0.3,
        )
        Cn, Cm = section_coefficients(sec)
        assert Cn == pytest.approx(0.0)
        assert Cm == pytest.approx(0.0)

    def test_single_point_returns_zero(self):
        sec = CpSection(x_over_c=np.array([0.5]),
                         delta_Cp=np.array([1.0]), eta=0.5)
        Cn, Cm = section_coefficients(sec)
        assert Cn == 0.0


# == integrate_cp_field =====================================================

def _flat_plate_field(alpha_deg=3.0, mach=0.6):
    """Synthetic CpField: uniform delta_Cp = 4*alpha/beta at 5 stations."""
    alpha = np.radians(alpha_deg)
    beta = np.sqrt(1.0 - mach ** 2)
    dCp = 4.0 * alpha / beta
    x = np.linspace(0, 1, 50)
    sections = [
        CpSection(x_over_c=x, delta_Cp=np.full(50, dCp), eta=e)
        for e in [0.1, 0.3, 0.5, 0.7, 0.9]
    ]
    q = 0.5 * 1.225 * (mach * 340.0) ** 2
    return CpField(sections=sections, mach=mach, alpha_deg=alpha_deg,
                   q_inf=q, source="synthetic")


class TestIntegrateCpField:
    def test_returns_correct_count(self):
        cp = _flat_plate_field()
        db = integrate_cp_field(
            cp, semi_span=5.0,
            chord_at_eta=lambda e: 2.0 * (1 - 0.3 * e),
            box_height_at_eta=lambda e: 0.04 * 2.0 * (1 - 0.3 * e),
            box_chord_at_eta=lambda e: 0.55 * 2.0 * (1 - 0.3 * e),
            sweep_deg_at_eta=lambda e: 30.0,
        )
        assert len(db) == 5  # one per section

    def test_nxx_is_compression(self):
        """Inboard stations should see compression; tip may be ~0."""
        cp = _flat_plate_field()
        db = integrate_cp_field(
            cp, semi_span=5.0,
            chord_at_eta=lambda e: 2.0,
            box_height_at_eta=lambda e: 0.08,
            box_chord_at_eta=lambda e: 1.1,
            sweep_deg_at_eta=lambda e: 30.0,
        )
        for case in db:
            assert case.Nxx <= 0, f"Expected compression, got Nxx={case.Nxx}"
        # Root should have meaningful compression
        assert db[0].Nxx < -100

    def test_nxx_decreases_toward_tip(self):
        cp = _flat_plate_field()
        db = integrate_cp_field(
            cp, semi_span=5.0,
            chord_at_eta=lambda e: 2.0 * (1 - 0.5 * e),
            box_height_at_eta=lambda e: 0.04 * 2.0 * (1 - 0.5 * e),
            box_chord_at_eta=lambda e: 0.55 * 2.0 * (1 - 0.5 * e),
            sweep_deg_at_eta=lambda e: 30.0,
        )
        # Tip should have less |Nxx| than root
        assert abs(db[-1].Nxx) < abs(db[0].Nxx)

    def test_source_propagated(self):
        cp = _flat_plate_field()
        db = integrate_cp_field(
            cp, semi_span=5.0,
            chord_at_eta=lambda e: 2.0,
            box_height_at_eta=lambda e: 0.08,
            box_chord_at_eta=lambda e: 1.1,
            sweep_deg_at_eta=lambda e: 0.0,
        )
        assert "synthetic" in db[0].source


# == Loaders ================================================================

class TestLoadCpCsv:
    def test_load_onera_m6(self):
        path = os.path.join(SCRIPTS, "onera_m6_pressure_points.csv")
        if not os.path.exists(path):
            pytest.skip("ONERA M6 CSV not found")
        cp = load_cp_csv(path, mach=0.8395, alpha_deg=3.06, q_inf=155883.0)
        assert len(cp) >= 5
        assert cp.mach == pytest.approx(0.8395)

    def test_sections_sorted_by_eta(self):
        path = os.path.join(SCRIPTS, "onera_m6_pressure_points.csv")
        if not os.path.exists(path):
            pytest.skip("ONERA M6 CSV not found")
        cp = load_cp_csv(path, mach=0.8395, alpha_deg=3.06, q_inf=155883.0)
        etas = [s.eta for s in cp.sections]
        assert etas == sorted(etas)


class TestLoadCpTecplot:
    def test_load_onera_tec(self):
        path = os.path.join(SCRIPTS, "ONERAb114.tec")
        if not os.path.exists(path):
            pytest.skip("ONERA .tec file not found")
        cp = load_cp_tecplot(path, q_inf=155883.0)
        assert len(cp) >= 5
        # Mach should be parsed from title
        assert cp.mach == pytest.approx(0.8395, abs=0.01)

    def test_sections_have_valid_delta_cp(self):
        path = os.path.join(SCRIPTS, "ONERAb114.tec")
        if not os.path.exists(path):
            pytest.skip("ONERA .tec file not found")
        cp = load_cp_tecplot(path, q_inf=155883.0)
        for sec in cp.sections:
            Cn, _ = section_coefficients(sec)
            assert Cn > 0, f"Expected positive Cn at eta={sec.eta}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
