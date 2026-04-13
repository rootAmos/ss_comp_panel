"""
tests/test_arrow_wing.py
Arrow-Wing CR-132575 loads database tests.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import math
import numpy as np
import pytest

from composite_panel.arrow_wing import (
    ArrowWingGeometry,
    arrow_wing_panel_loads,
    arrow_wing_loads_database,
    FLIGHT_CONDITIONS,
    SPAN_STATIONS,
)
from composite_panel.loads_db import LoadsDatabase


# == Geometry ==============================================================

class TestArrowWingGeometry:
    def test_defaults_match_cr132575(self):
        g = ArrowWingGeometry()
        assert g.semi_span == pytest.approx(22.25)
        assert g.root_chord == pytest.approx(38.1)
        assert g.togw_kg == pytest.approx(340_194)
        assert g.sweep_inboard_deg == pytest.approx(74.0)
        assert g.sweep_outboard_deg == pytest.approx(57.0)
        assert g.t_over_c == pytest.approx(0.03)

    def test_chord_root(self):
        g = ArrowWingGeometry()
        assert g.chord(0.0) == pytest.approx(g.root_chord)

    def test_chord_kink(self):
        g = ArrowWingGeometry()
        assert g.chord(g.kink_eta) == pytest.approx(g.kink_chord)

    def test_chord_tip(self):
        g = ArrowWingGeometry()
        assert g.chord(1.0) == pytest.approx(g.tip_chord)

    def test_chord_monotonic_decreasing(self):
        g = ArrowWingGeometry()
        etas = np.linspace(0, 1, 50)
        chords = [g.chord(e) for e in etas]
        for i in range(1, len(chords)):
            assert chords[i] <= chords[i - 1] + 1e-10

    def test_sweep_inboard(self):
        g = ArrowWingGeometry()
        assert g.sweep_deg(0.20) == 74.0

    def test_sweep_outboard(self):
        g = ArrowWingGeometry()
        assert g.sweep_deg(0.60) == 57.0

    def test_sweep_at_kink(self):
        """Kink station itself belongs to inboard panel."""
        g = ArrowWingGeometry()
        assert g.sweep_deg(g.kink_eta) == g.sweep_inboard_deg

    def test_weight_n(self):
        g = ArrowWingGeometry()
        assert g.weight_n == pytest.approx(340_194 * 9.80665, rel=1e-6)

    def test_box_height(self):
        g = ArrowWingGeometry()
        assert g.box_height(0.0) == pytest.approx(g.t_over_c * g.root_chord)

    def test_box_chord(self):
        g = ArrowWingGeometry()
        assert g.box_chord(0.0) == pytest.approx(g.box_chord_frac * g.root_chord)


# == Panel loads ===========================================================

class TestArrowWingPanelLoads:
    def test_supersonic_compression_upper_skin(self):
        """At positive n, upper skin should see spanwise compression (Nxx < 0)."""
        case = arrow_wing_panel_loads(
            eta=0.35, mach=2.7, altitude_m=19_800,
            alpha_deg=7.5, n_load=2.5,
        )
        assert case.Nxx < 0, "Expected Nxx < 0 (compression)"

    def test_nyy_from_pressure_only(self):
        """In spar frame, Nyy is pressure only -- small relative to Nxx."""
        case = arrow_wing_panel_loads(
            eta=0.35, mach=2.7, altitude_m=19_800,
            alpha_deg=7.5, n_load=2.5,
        )
        assert case.Nyy < 0, "Expected Nyy < 0 (pressure compression)"
        assert abs(case.Nyy) < abs(case.Nxx), (
            "In spar-aligned frame, |Nyy| (pressure only) should be << |Nxx| (bending)"
        )

    def test_shear_positive(self):
        case = arrow_wing_panel_loads(
            eta=0.35, mach=2.7, altitude_m=19_800,
            alpha_deg=7.5, n_load=2.5,
        )
        assert case.Nxy > 0, "Expected positive shear from V and T"

    def test_nxx_dominates_in_spar_frame(self):
        """In spar-aligned frame, bending compression (Nxx) is the primary load."""
        case = arrow_wing_panel_loads(
            eta=0.35, mach=2.7, altitude_m=19_800,
            alpha_deg=7.5, n_load=2.5,
        )
        assert abs(case.Nxx) > abs(case.Nyy), (
            f"|Nxx|={abs(case.Nxx) / 1e3:.0f} should exceed "
            f"|Nyy|={abs(case.Nyy) / 1e3:.0f} kN/m in spar frame"
        )

    def test_pushover_gives_tension(self):
        """Negative n -> upper skin in tension (Nxx > 0)."""
        case = arrow_wing_panel_loads(
            eta=0.35, mach=2.7, altitude_m=19_800,
            alpha_deg=-3.0, n_load=-1.0,
        )
        assert case.Nxx > 0, "Expected Nxx > 0 (tension) in pushover"

    def test_loads_small_near_tip(self):
        """Near-tip bending moment is small, so loads should be modest."""
        mid = arrow_wing_panel_loads(
            eta=0.50, mach=2.7, altitude_m=19_800,
            alpha_deg=7.5, n_load=2.5,
        )
        tip = arrow_wing_panel_loads(
            eta=0.90, mach=2.7, altitude_m=19_800,
            alpha_deg=7.5, n_load=2.5,
        )
        assert abs(tip.Nxx) < abs(mid.Nxx)

    def test_subsonic_works(self):
        case = arrow_wing_panel_loads(
            eta=0.50, mach=0.6, altitude_m=3_048,
            alpha_deg=5.0, n_load=1.8,
        )
        assert case.Nxx < 0

    def test_transonic_raises(self):
        with pytest.raises(ValueError, match="transonic"):
            arrow_wing_panel_loads(
                eta=0.35, mach=1.0, altitude_m=10_000, alpha_deg=5.0,
            )

    def test_eta_recorded(self):
        case = arrow_wing_panel_loads(
            eta=0.70, mach=2.7, altitude_m=19_800,
            alpha_deg=3.0, n_load=1.0,
        )
        assert case.eta == pytest.approx(0.70)

    def test_source_identifies_method(self):
        case = arrow_wing_panel_loads(
            eta=0.35, mach=2.7, altitude_m=19_800,
            alpha_deg=3.0, n_load=1.0,
        )
        assert "strip-theory" in case.source
        assert "Ackeret" in case.source  # supersonic regime

    def test_mxx_positive(self):
        """Pressure bending moment should be positive."""
        case = arrow_wing_panel_loads(
            eta=0.35, mach=2.7, altitude_m=19_800,
            alpha_deg=7.5, n_load=2.5,
        )
        assert case.Mxx > 0


# == Database factory ======================================================

class TestArrowWingDatabase:
    def test_default_case_count(self):
        db = arrow_wing_loads_database()
        expected = len(FLIGHT_CONDITIONS) * len(SPAN_STATIONS)
        assert len(db) == expected

    def test_csv_roundtrip(self, tmp_path):
        db = arrow_wing_loads_database()
        csv_path = str(tmp_path / "arrow_wing.csv")
        db.to_csv(csv_path)
        db2 = LoadsDatabase.from_csv(csv_path)
        assert len(db2) == len(db)
        assert abs(db2[0].Nxx - db[0].Nxx) < 1.0

    def test_envelope_case(self):
        db = arrow_wing_loads_database()
        env = db.envelope_case()
        assert env.Nxx < 0, "Envelope should include compression"
        assert env.Nxy != 0, "Envelope should include shear"

    def test_filter_eta(self):
        db = arrow_wing_loads_database()
        mid = db.filter_eta(0.50)
        assert len(mid) == len(FLIGHT_CONDITIONS)

    def test_filter_source(self):
        db = arrow_wing_loads_database()
        st = db.filter_source("strip-theory")
        assert len(st) == len(db)

    def test_custom_geometry(self):
        geom = ArrowWingGeometry(togw_kg=200_000)
        db = arrow_wing_loads_database(geom=geom)
        db_default = arrow_wing_loads_database()
        assert len(db) == len(db_default)
        # Lighter aircraft -> smaller |Nxx| at same station/condition
        assert abs(db[0].Nxx) < abs(db_default[0].Nxx)

    def test_summary_runs(self):
        db = arrow_wing_loads_database()
        s = db.summary()
        assert "LoadsDatabase" in s
        assert "35 cases" in s


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
