"""
composite_panel.loads_store
---------------------------
Scalable loads database backed by SQLite.

Provides indexed storage and rich querying across load sets, flight
conditions, and span stations.  Designed to grow from prototype through
production -- from dozens of computed cases to thousands of CFD/test points.

Key concepts:
    **LoadSet**  -- a versioned batch of load cases from one analysis source
                    (e.g. "baseline_CFD_v2", "wind_tunnel_2024Q3").
    **LoadCase** -- a single panel running-load state at one span station
                    and flight condition.

Usage
-----
    from composite_panel.loads_store import LoadsStore

    store = LoadsStore("project_loads.db")
    ls = store.create_load_set("baseline_CFD_v1", source="FUN3D")
    store.add_cases(my_cases, load_set_id=ls, mach=2.7, n_load=2.5)

    # Query
    critical = store.query(eta=0.35, mach_range=(1.5, 3.0))
    env = store.envelope(eta=0.35)
    critical.print_summary()

    # Export for sizing
    db = store.query(n_load_range=(2.0, 3.0))
    result = optimize_laminate_multicase(db.cases, mat, angles, ...)
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from typing import Any, List, Optional

from .loads_db import LoadCase, LoadsDatabase


# -----------------------------------------------------------------------
# Schema
# -----------------------------------------------------------------------

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS load_sets (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT UNIQUE NOT NULL,
    source      TEXT DEFAULT '',
    description TEXT DEFAULT '',
    config      TEXT DEFAULT '',
    version     TEXT DEFAULT '',
    created_at  TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS load_cases (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    load_set_id INTEGER REFERENCES load_sets(id),
    name        TEXT NOT NULL,
    Nxx         REAL NOT NULL,
    Nyy         REAL NOT NULL,
    Nxy         REAL NOT NULL,
    Mxx         REAL DEFAULT 0,
    Myy         REAL DEFAULT 0,
    Mxy         REAL DEFAULT 0,
    eta         REAL,
    mach        REAL,
    altitude_m  REAL,
    alpha_deg   REAL,
    n_load      REAL,
    source      TEXT DEFAULT '',
    description TEXT DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_lc_load_set ON load_cases(load_set_id);
CREATE INDEX IF NOT EXISTS idx_lc_eta      ON load_cases(eta);
CREATE INDEX IF NOT EXISTS idx_lc_mach     ON load_cases(mach);
CREATE INDEX IF NOT EXISTS idx_lc_n_load   ON load_cases(n_load);
"""


# -----------------------------------------------------------------------
# LoadsStore
# -----------------------------------------------------------------------

class LoadsStore:
    """SQLite-backed loads database with indexed queries.

    Parameters
    ----------
    path : str
        File path for the SQLite database.  Use ``":memory:"`` for an
        in-memory store (useful for tests and ephemeral pipelines).
    """

    def __init__(self, path: str = ":memory:"):
        self.path = path
        self._conn = sqlite3.connect(path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    def _init_schema(self):
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self):
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -------------------------------------------------------------------
    # Load sets
    # -------------------------------------------------------------------

    def create_load_set(
        self,
        name: str,
        *,
        source: str = "",
        description: str = "",
        config: dict | None = None,
        version: str = "",
    ) -> int:
        """Create a new load set.  Returns the load_set id."""
        config_json = json.dumps(config) if config else ""
        cur = self._conn.execute(
            "INSERT INTO load_sets (name, source, description, config, version) "
            "VALUES (?, ?, ?, ?, ?)",
            (name, source, description, config_json, version),
        )
        self._conn.commit()
        return cur.lastrowid

    def load_sets(self) -> list[dict]:
        """List all load sets with metadata and case counts."""
        rows = self._conn.execute(
            "SELECT ls.*, COUNT(lc.id) AS case_count "
            "FROM load_sets ls LEFT JOIN load_cases lc ON ls.id = lc.load_set_id "
            "GROUP BY ls.id ORDER BY ls.id"
        ).fetchall()
        return [dict(r) for r in rows]

    def delete_load_set(self, name_or_id) -> int:
        """Delete a load set and all its cases.  Returns cases deleted."""
        ls_id = self._resolve_load_set(name_or_id)
        if ls_id is None:
            return 0
        cur = self._conn.execute(
            "DELETE FROM load_cases WHERE load_set_id = ?", (ls_id,)
        )
        n = cur.rowcount
        self._conn.execute("DELETE FROM load_sets WHERE id = ?", (ls_id,))
        self._conn.commit()
        return n

    def _resolve_load_set(self, name_or_id) -> int | None:
        if isinstance(name_or_id, int):
            return name_or_id
        row = self._conn.execute(
            "SELECT id FROM load_sets WHERE name = ?", (name_or_id,)
        ).fetchone()
        return row["id"] if row else None

    # -------------------------------------------------------------------
    # Adding cases
    # -------------------------------------------------------------------

    def add_cases(
        self,
        cases: list[LoadCase] | LoadsDatabase,
        load_set_id: int | None = None,
        *,
        mach: float | None = None,
        altitude_m: float | None = None,
        alpha_deg: float | None = None,
        n_load: float | None = None,
    ) -> int:
        """Bulk-insert LoadCase objects.  Returns count inserted.

        Flight-condition metadata (mach, altitude, etc.) can be supplied
        per-batch here; individual LoadCase fields override when present.
        """
        if isinstance(cases, LoadsDatabase):
            cases = cases.cases

        rows = []
        for c in cases:
            rows.append((
                load_set_id,
                c.name,
                c.Nxx, c.Nyy, c.Nxy,
                c.Mxx, c.Myy, c.Mxy,
                c.eta if c.eta == c.eta else None,  # nan -> NULL
                mach, altitude_m, alpha_deg, n_load,
                c.source, c.description,
            ))

        self._conn.executemany(
            "INSERT INTO load_cases "
            "(load_set_id, name, Nxx, Nyy, Nxy, Mxx, Myy, Mxy, "
            " eta, mach, altitude_m, alpha_deg, n_load, source, description) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            rows,
        )
        self._conn.commit()
        return len(rows)

    def ingest_csv(
        self,
        path: str,
        load_set_id: int | None = None,
        **metadata,
    ) -> int:
        """Import a CSV file in LoadsDatabase format.  Returns count."""
        db = LoadsDatabase.from_csv(path)
        return self.add_cases(db, load_set_id, **metadata)

    # -------------------------------------------------------------------
    # Querying
    # -------------------------------------------------------------------

    def query(
        self,
        *,
        load_set: str | int | None = None,
        mach: float | None = None,
        mach_range: tuple[float, float] | None = None,
        eta: float | None = None,
        eta_tol: float = 0.02,
        eta_range: tuple[float, float] | None = None,
        n_load: float | None = None,
        n_load_range: tuple[float, float] | None = None,
        altitude_range: tuple[float, float] | None = None,
        source: str | None = None,
        limit: int | None = None,
    ) -> LoadsDatabase:
        """Query load cases with optional filters.  Returns a LoadsDatabase."""
        clauses: list[str] = []
        params: list[Any] = []

        if load_set is not None:
            ls_id = self._resolve_load_set(load_set)
            clauses.append("load_set_id = ?")
            params.append(ls_id)

        if mach is not None:
            clauses.append("abs(mach - ?) < 0.001")
            params.append(mach)
        if mach_range is not None:
            clauses.append("mach >= ? AND mach <= ?")
            params.extend(mach_range)

        if eta is not None:
            clauses.append("abs(eta - ?) <= ?")
            params.extend([eta, eta_tol])
        if eta_range is not None:
            clauses.append("eta >= ? AND eta <= ?")
            params.extend(eta_range)

        if n_load is not None:
            clauses.append("abs(n_load - ?) < 0.01")
            params.append(n_load)
        if n_load_range is not None:
            clauses.append("n_load >= ? AND n_load <= ?")
            params.extend(n_load_range)

        if altitude_range is not None:
            clauses.append("altitude_m >= ? AND altitude_m <= ?")
            params.extend(altitude_range)

        if source is not None:
            clauses.append("source LIKE ?")
            params.append(f"%{source}%")

        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = f"SELECT * FROM load_cases{where} ORDER BY eta, mach"
        if limit is not None:
            sql += f" LIMIT {int(limit)}"

        rows = self._conn.execute(sql, params).fetchall()
        return LoadsDatabase([self._row_to_case(r) for r in rows])

    def envelope(self, **filters) -> LoadCase:
        """Compute the component-wise envelope across filtered cases."""
        db = self.query(**filters)
        return db.envelope_case()

    def count(self, **filters) -> int:
        """Count cases matching filters (same kwargs as query)."""
        return len(self.query(**filters))

    @staticmethod
    def _row_to_case(row: sqlite3.Row) -> LoadCase:
        return LoadCase(
            name=row["name"],
            Nxx=row["Nxx"], Nyy=row["Nyy"], Nxy=row["Nxy"],
            Mxx=row["Mxx"] or 0.0,
            Myy=row["Myy"] or 0.0,
            Mxy=row["Mxy"] or 0.0,
            source=row["source"] or "",
            eta=row["eta"] if row["eta"] is not None else float("nan"),
            description=row["description"] or "",
        )

    # -------------------------------------------------------------------
    # Summary / diagnostics
    # -------------------------------------------------------------------

    def summary(self) -> str:
        """Overview of the store contents."""
        total = self._conn.execute(
            "SELECT COUNT(*) FROM load_cases"
        ).fetchone()[0]

        lines = [f"LoadsStore  --  {total} cases total"]

        for ls in self.load_sets():
            lines.append(
                f"  [{ls['id']}] {ls['name']:<30} "
                f"{ls['case_count']:>5} cases  "
                f"src={ls['source']!r}"
            )

        # Mach / eta / n_load ranges
        stats = self._conn.execute(
            "SELECT MIN(mach), MAX(mach), MIN(eta), MAX(eta), "
            "       MIN(n_load), MAX(n_load) "
            "FROM load_cases"
        ).fetchone()
        if stats and stats[0] is not None:
            lines.append(
                f"  Mach {stats[0]:.1f}-{stats[1]:.1f}  "
                f"eta {stats[2]:.2f}-{stats[3]:.2f}  "
                f"n {stats[4]:.1f}-{stats[5]:.1f}g"
            )

        return "\n".join(lines)

    def print_summary(self) -> None:
        print(self.summary())
