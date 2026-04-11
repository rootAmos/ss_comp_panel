"""
tests/test_input_parsing.py
Input parsing: messy CSVs, mixed units, key aliases, error handling.
"""
import sys, os, io, textwrap, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import math
import numpy as np
import pytest

from composite_panel.loads_db import LoadCase, LoadsDatabase
from composite_panel.ply import PlyMaterial


# -- helpers -------------------------------------------------------------------

def _write_csv(content: str) -> str:
    """Write content to a temp file, return path."""
    f = tempfile.NamedTemporaryFile(mode='w', suffix='.csv',
                                    delete=False, encoding='utf-8')
    f.write(textwrap.dedent(content))
    f.close()
    return f.name


# ===============================================================================
# LoadsDatabase.from_csv  --  messy CSV
# ===============================================================================

def test_clean_csv_roundtrip(tmp_path):
    db = LoadsDatabase([
        LoadCase("A", -100e3, -40e3,  5e3),
        LoadCase("B", -200e3, -80e3, 10e3, eta=0.45),
    ])
    p = str(tmp_path / "out.csv")
    db.to_csv(p)
    db2 = LoadsDatabase.from_csv(p)
    assert len(db2) == 2
    assert abs(db2[0].Nxx - db[0].Nxx) < 1.0
    assert abs(db2[1].eta - 0.45) < 1e-4


def test_case_insensitive_headers():
    path = _write_csv("""\
        NXX,NYY,NXY,name
        -100000,-40000,5000,case1
    """)
    db = LoadsDatabase.from_csv(path)
    assert len(db) == 1
    assert abs(db[0].Nxx - (-100e3)) < 1.0


def test_headers_with_spaces():
    path = _write_csv("""\
        name , N xx , N yy , N xy
        load1,-150000,-50000,8000
    """)
    # "N xx" normalises to "n_xx" -> alias -> Nxx
    db = LoadsDatabase.from_csv(path)
    assert abs(db[0].Nxx - (-150e3)) < 1.0


def test_kN_per_m_suffix_conversion():
    path = _write_csv("""\
        name,Nxx,Nyy,Nxy
        case1,"-150 kN/m","-50 kN/m","8 kN/m"
    """)
    db = LoadsDatabase.from_csv(path)
    assert abs(db[0].Nxx - (-150e3)) < 1.0
    assert abs(db[0].Nyy - (-50e3))  < 1.0
    assert abs(db[0].Nxy -   8e3)    < 1.0


def test_parenthesised_negative():
    path = _write_csv("""\
        name,Nxx,Nyy,Nxy
        case1,(150000),(50000),8000
    """)
    db = LoadsDatabase.from_csv(path)
    assert db[0].Nxx == pytest.approx(-150e3)
    assert db[0].Nyy == pytest.approx(-50e3)


def test_thousands_separator():
    path = _write_csv("""\
        name,Nxx,Nyy,Nxy
        case1,"-1,500,000",-50000,8000
    """)
    db = LoadsDatabase.from_csv(path)
    assert db[0].Nxx == pytest.approx(-1.5e6)


def test_comment_lines_skipped():
    path = _write_csv("""\
        # this is a comment
        name,Nxx,Nyy,Nxy
        # another comment
        case1,-100000,-40000,5000
    """)
    db = LoadsDatabase.from_csv(path)
    assert len(db) == 1


def test_missing_optional_columns_default_zero():
    path = _write_csv("""\
        name,Nxx,Nyy,Nxy
        case1,-100000,-40000,5000
    """)
    db = LoadsDatabase.from_csv(path)
    assert db[0].Mxx == 0.0
    assert db[0].Myy == 0.0
    assert math.isnan(db[0].eta)


def test_extra_unknown_columns_ignored():
    path = _write_csv("""\
        name,Nxx,Nyy,Nxy,flight_phase,analyst
        case1,-100000,-40000,5000,cruise,JD
    """)
    db = LoadsDatabase.from_csv(path)
    assert len(db) == 1
    assert db[0].Nxx == pytest.approx(-100e3)


def test_missing_required_column_raises():
    path = _write_csv("""\
        name,Nxx,Nyy
        case1,-100000,-40000
    """)
    with pytest.raises(ValueError, match="missing required"):
        LoadsDatabase.from_csv(path)


def test_unparseable_value_raises():
    path = _write_csv("""\
        name,Nxx,Nyy,Nxy
        case1,NOT_A_NUMBER,-40000,5000
    """)
    with pytest.raises(ValueError, match="Cannot parse"):
        LoadsDatabase.from_csv(path)


# ===============================================================================
# LoadsDatabase.from_dict
# ===============================================================================

def test_from_dict_basic():
    db = LoadsDatabase.from_dict([
        {"name": "1g_cruise", "Nxx": -120e3, "Nyy": -40e3, "Nxy": 5e3},
        {"name": "3g_pull",   "Nxx": -400e3, "Nyy": -60e3, "Nxy": 12e3},
    ])
    assert len(db) == 2
    assert db[0].Nxx == pytest.approx(-120e3)


def test_from_dict_case_insensitive_keys():
    db = LoadsDatabase.from_dict([
        {"NAME": "A", "NXX": -100e3, "NYY": -30e3, "NXY": 5e3},
    ])
    assert db[0].name == "A"
    assert db[0].Nxx == pytest.approx(-100e3)


def test_from_dict_kN_string_values():
    db = LoadsDatabase.from_dict([
        {"name": "A", "Nxx": "-150 kN/m", "Nyy": "-50 kN/m", "Nxy": "8 kN/m"},
    ])
    assert db[0].Nxx == pytest.approx(-150e3)


def test_from_dict_missing_required_raises():
    with pytest.raises(ValueError, match="missing required"):
        LoadsDatabase.from_dict([{"name": "A", "Nxx": -100e3}])


# ===============================================================================
# PlyMaterial.from_dict
# ===============================================================================

def test_from_dict_eng_units():
    mat = PlyMaterial.from_dict({
        "E1": 171.4, "E2": 9.08, "G12": 5.29, "nu12": 0.32,
    }, units="eng")
    assert mat.E1 == pytest.approx(171.4e9, rel=1e-6)
    assert mat.E2 == pytest.approx(9.08e9,  rel=1e-6)


def test_from_dict_SI_passthrough():
    mat = PlyMaterial.from_dict({
        "E1": 171.4e9, "E2": 9.08e9, "G12": 5.29e9, "nu12": 0.32,
    })
    assert mat.E1 == pytest.approx(171.4e9, rel=1e-6)


def test_from_dict_eng_strength():
    mat = PlyMaterial.from_dict({
        "E1": 171.4, "E2": 9.08, "G12": 5.29, "nu12": 0.32,
        "F1t": 2326, "F1c": 1200, "F2t": 62, "F2c": 200, "F12": 110,
    }, units="eng")
    assert mat.F1t == pytest.approx(2326e6, rel=1e-6)
    assert mat.F2t == pytest.approx(62e6,   rel=1e-6)


def test_from_dict_key_aliases():
    mat = PlyMaterial.from_dict({
        "e_fibre": 171.4, "e_transverse": 9.08, "g_shear": 5.29, "poisson": 0.32,
        "Xt": 2326, "Xc": 1200, "Yt": 62, "Yc": 200, "S12": 110,
    }, units="eng")
    assert mat.E1  == pytest.approx(171.4e9, rel=1e-6)
    assert mat.F1t == pytest.approx(2326e6,  rel=1e-6)


def test_from_dict_maxwell_reciprocity():
    mat = PlyMaterial.from_dict({
        "E1": 171.4e9, "E2": 9.08e9, "G12": 5.29e9, "nu12": 0.32,
    })
    assert mat.nu21 == pytest.approx(mat.nu12 * mat.E2 / mat.E1, rel=1e-10)


def test_from_dict_missing_required_raises():
    with pytest.raises(ValueError, match="missing required"):
        PlyMaterial.from_dict({"E1": 171.4e9, "E2": 9.08e9})  # missing G12, nu12


def test_from_dict_matches_IM7_8552():
    from composite_panel import IM7_8552
    ref = IM7_8552()
    mat = PlyMaterial.from_dict({
        "E1": ref.E1 / 1e9, "E2": ref.E2 / 1e9,
        "G12": ref.G12 / 1e9, "nu12": ref.nu12,
        "F1t": ref.F1t / 1e6, "F1c": ref.F1c / 1e6,
        "F2t": ref.F2t / 1e6, "F2c": ref.F2c / 1e6,
        "F12": ref.F12 / 1e6,
    }, units="eng")
    assert mat.E1  == pytest.approx(ref.E1,  rel=1e-6)
    assert mat.F1t == pytest.approx(ref.F1t, rel=1e-6)


def test_from_dict_small_values_raise():
    with pytest.raises(ValueError, match="too small"):
        PlyMaterial.from_dict({
            "E1": 171.4, "E2": 9.08, "G12": 5.29, "nu12": 0.32,
        })  # forgot units='eng' -- should fail, not silently convert
