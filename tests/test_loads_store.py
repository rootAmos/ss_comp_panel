"""
tests/test_loads_store.py
Scalable loads store tests.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest

from composite_panel.loads_db import LoadCase, LoadsDatabase
from composite_panel.loads_store import LoadsStore

SCRIPTS = os.path.join(os.path.dirname(__file__), '..', 'scripts')


def _sample_cases():
    return [
        LoadCase("M2_2g_root", -300e3, -1e3, 150e3, source="CFD", eta=0.15),
        LoadCase("M2_2g_mid",  -500e3, -2e3, 200e3, source="CFD", eta=0.50),
        LoadCase("M2_2g_tip",  -100e3, -0.5e3, 50e3, source="CFD", eta=0.90),
    ]


# == Basic operations =======================================================

class TestLoadSetCRUD:
    def test_create_load_set(self):
        with LoadsStore() as store:
            ls_id = store.create_load_set("test_v1", source="CFD")
            assert ls_id >= 1

    def test_list_load_sets(self):
        with LoadsStore() as store:
            store.create_load_set("a", source="CFD")
            store.create_load_set("b", source="WT")
            sets = store.load_sets()
            assert len(sets) == 2
            assert sets[0]["name"] == "a"

    def test_duplicate_name_raises(self):
        with LoadsStore() as store:
            store.create_load_set("dup")
            with pytest.raises(Exception):
                store.create_load_set("dup")

    def test_delete_load_set(self):
        with LoadsStore() as store:
            ls = store.create_load_set("temp")
            store.add_cases(_sample_cases(), ls, mach=2.0, n_load=2.5)
            n = store.delete_load_set("temp")
            assert n == 3
            assert store.count() == 0


# == Adding and querying cases ==============================================

class TestAddAndQuery:
    def test_add_cases(self):
        with LoadsStore() as store:
            ls = store.create_load_set("run1")
            n = store.add_cases(_sample_cases(), ls, mach=2.0, n_load=2.5)
            assert n == 3
            assert store.count() == 3

    def test_query_all(self):
        with LoadsStore() as store:
            ls = store.create_load_set("run1")
            store.add_cases(_sample_cases(), ls, mach=2.0, n_load=2.5)
            db = store.query()
            assert len(db) == 3

    def test_query_by_eta(self):
        with LoadsStore() as store:
            ls = store.create_load_set("run1")
            store.add_cases(_sample_cases(), ls, mach=2.0, n_load=2.5)
            db = store.query(eta=0.50)
            assert len(db) == 1
            assert abs(db[0].eta - 0.50) < 0.03

    def test_query_by_mach_range(self):
        with LoadsStore() as store:
            ls1 = store.create_load_set("sub")
            ls2 = store.create_load_set("sup")
            store.add_cases(_sample_cases(), ls1, mach=0.8, n_load=2.5)
            store.add_cases(_sample_cases(), ls2, mach=2.7, n_load=2.5)
            db = store.query(mach_range=(1.0, 3.0))
            assert len(db) == 3  # only supersonic

    def test_query_by_n_load_range(self):
        with LoadsStore() as store:
            ls = store.create_load_set("run1")
            store.add_cases(_sample_cases(), ls, mach=2.0, n_load=2.5)
            store.add_cases(_sample_cases(), ls, mach=2.0, n_load=1.0)
            db = store.query(n_load_range=(2.0, 3.0))
            assert len(db) == 3

    def test_query_by_load_set(self):
        with LoadsStore() as store:
            ls1 = store.create_load_set("CFD_v1")
            ls2 = store.create_load_set("CFD_v2")
            store.add_cases(_sample_cases(), ls1, mach=2.0, n_load=2.5)
            store.add_cases(_sample_cases(), ls2, mach=2.0, n_load=2.5)
            db = store.query(load_set="CFD_v1")
            assert len(db) == 3

    def test_query_by_source_substring(self):
        with LoadsStore() as store:
            ls = store.create_load_set("run1")
            store.add_cases(_sample_cases(), ls, mach=2.0, n_load=2.5)
            db = store.query(source="CFD")
            assert len(db) == 3

    def test_query_with_limit(self):
        with LoadsStore() as store:
            ls = store.create_load_set("run1")
            store.add_cases(_sample_cases(), ls, mach=2.0, n_load=2.5)
            db = store.query(limit=2)
            assert len(db) == 2

    def test_envelope(self):
        with LoadsStore() as store:
            ls = store.create_load_set("run1")
            store.add_cases(_sample_cases(), ls, mach=2.0, n_load=2.5)
            env = store.envelope()
            assert env.Nxx == pytest.approx(-500e3)
            assert env.Nxy == pytest.approx(200e3)


# == CSV ingestion ===========================================================

class TestIngestCSV:
    def test_ingest_flight_envelope(self):
        path = os.path.join(SCRIPTS, "flight_envelope.csv")
        if not os.path.exists(path):
            pytest.skip("flight_envelope.csv not found")
        with LoadsStore() as store:
            ls = store.create_load_set("flight_env", source="computed")
            n = store.ingest_csv(path, ls)
            assert n >= 20
            db = store.query()
            assert len(db) == n

    def test_ingest_published_loads(self):
        path = os.path.join(SCRIPTS, "published_panel_loads.csv")
        if not os.path.exists(path):
            pytest.skip("published_panel_loads.csv not found")
        with LoadsStore() as store:
            ls = store.create_load_set("published", source="literature")
            n = store.ingest_csv(path, ls)
            assert n >= 4


# == Persistence =============================================================

class TestPersistence:
    def test_file_roundtrip(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        with LoadsStore(db_path) as store:
            ls = store.create_load_set("run1")
            store.add_cases(_sample_cases(), ls, mach=2.0, n_load=2.5)

        # Reopen
        with LoadsStore(db_path) as store2:
            assert store2.count() == 3
            sets = store2.load_sets()
            assert sets[0]["name"] == "run1"

    def test_summary(self):
        with LoadsStore() as store:
            ls = store.create_load_set("run1", source="CFD")
            store.add_cases(_sample_cases(), ls, mach=2.0, n_load=2.5)
            s = store.summary()
            assert "3 cases" in s
            assert "run1" in s


# == Integration with downstream ============================================

class TestDownstreamCompat:
    def test_query_returns_loads_database(self):
        with LoadsStore() as store:
            ls = store.create_load_set("run1")
            store.add_cases(_sample_cases(), ls, mach=2.0, n_load=2.5)
            db = store.query()
            assert isinstance(db, LoadsDatabase)
            # Can call LoadsDatabase methods
            env = db.envelope_case()
            assert env.Nxx < 0
            s = db.summary()
            assert "3 cases" in s


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
