"""
composite_panel.loads_db
------------------------
Load case container and CSV-backed database for multi-condition sizing.

CSV schema: name, Nxx, Nyy, Nxy [required],
            Mxx, Myy, Mxy, source, eta, description [optional]
Units: N/m (running forces), N.m/m (moments).  Compression negative.

Refs: CS-25 §25.305, MIL-A-8861
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field, asdict
from typing import ClassVar, List, Optional, Iterator
import numpy as _np




@dataclass
class LoadCase:
    """
    Panel running loads at a single flight condition.

    All loads follow the CLT sign convention:
      - Compression is negative for Nxx, Nyy
      - Positive Nxy is right-hand shear on the x-face

    Attributes
    ----------
    name        : identifier (e.g. 'M1.7_2.5g_root')
    Nxx         : spanwise running force [N/m]
    Nyy         : chordwise running force [N/m]
    Nxy         : in-plane shear force [N/m]
    Mxx         : spanwise bending moment [N·m/m]
    Myy         : chordwise bending moment [N·m/m]
    Mxy         : twisting moment [N·m/m]
    source      : origin of loads (e.g. 'CFD', 'Ackeret', 'VLM', 'test')
    eta         : spanwise station η = y/b  (optional; for sorting/filtering)
    description : free-text annotation
    """
    name:        str
    Nxx:         float
    Nyy:         float
    Nxy:         float
    Mxx:         float = 0.0
    Myy:         float = 0.0
    Mxy:         float = 0.0
    source:      str   = ""
    eta:         float = float("nan")
    description: str   = ""


    @property
    def N(self) -> _np.ndarray:
        """Force resultant vector [Nxx, Nyy, Nxy] [N/m]."""
        return _np.array([self.Nxx, self.Nyy, self.Nxy], dtype=float)

    @property
    def M(self) -> _np.ndarray:
        """Moment resultant vector [Mxx, Myy, Mxy] [N·m/m]."""
        return _np.array([self.Mxx, self.Myy, self.Mxy], dtype=float)

    @property
    def max_compression(self) -> float:
        """Most compressive running force (min of Nxx, Nyy) [N/m]."""
        return min(self.Nxx, self.Nyy)

    def scaled(self, factor: float) -> "LoadCase":
        """Return a new LoadCase with all loads multiplied by factor."""
        return LoadCase(
            name        = f"{self.name}_x{factor:.2f}",
            Nxx         = self.Nxx * factor,
            Nyy         = self.Nyy * factor,
            Nxy         = self.Nxy * factor,
            Mxx         = self.Mxx * factor,
            Myy         = self.Myy * factor,
            Mxy         = self.Mxy * factor,
            source      = self.source,
            eta         = self.eta,
            description = self.description,
        )

    def __str__(self) -> str:
        return (f"LoadCase({self.name!r}  "
                f"Nxx={self.Nxx/1e3:+.1f}  Nyy={self.Nyy/1e3:+.1f}  "
                f"Nxy={self.Nxy/1e3:+.1f} kN/m  src={self.source!r})")




class LoadsDatabase:
    """
    Ordered collection of LoadCase objects with CSV I/O.

    Usage
    -----
    >>> db = LoadsDatabase.from_csv("wing_envelope.csv")
    >>> db.summary()
    >>> result = optimize_laminate_multicase(db.cases, mat, angles, ...)
    """

    def __init__(self, cases: List[LoadCase]):
        self.cases: List[LoadCase] = list(cases)

    def __len__(self) -> int:
        return len(self.cases)

    def __iter__(self) -> Iterator[LoadCase]:
        return iter(self.cases)

    def __getitem__(self, idx):
        return self.cases[idx]

    def append(self, case: LoadCase) -> None:
        self.cases.append(case)


    _REQUIRED = ("name", "Nxx", "Nyy", "Nxy")
    _FLOAT_FIELDS = ("Nxx", "Nyy", "Nxy", "Mxx", "Myy", "Mxy", "eta")

    # Aliases: maps lowercase variant → canonical field name.
    # Handles column headers from different tools/conventions.
    _ALIASES: ClassVar[dict] = {
        "n_xx": "Nxx", "nxx": "Nxx", "nx": "Nxx", "nspan": "Nxx",
        "n_yy": "Nyy", "nyy": "Nyy", "ny": "Nyy", "nchord": "Nyy",
        "n_xy": "Nxy", "nxy": "Nxy", "nshear": "Nxy",
        "m_xx": "Mxx", "mxx": "Mxx", "mx": "Mxx",
        "m_yy": "Myy", "myy": "Myy", "my": "Myy",
        "m_xy": "Mxy", "mxy": "Mxy",
        "case": "name", "case_name": "name", "id": "name", "load_case": "name",
        "span_station": "eta", "y_over_b": "eta", "y/b": "eta",
        "src": "source", "origin": "source",
        "desc": "description", "note": "description", "notes": "description",
    }

    @staticmethod
    def _parse_float(raw: str, field: str, row_num: int) -> float:
        """
        Parse a float from a raw CSV string, handling common real-world formats:
          - Whitespace padding:   "  -150.0  "
          - kN/m suffix:          "-150 kN/m"  → multiplied by 1000
          - kN suffix:            "-150kN"     → multiplied by 1000
          - Parenthesised negatives: "(150)"   → -150
          - Comma thousands sep:  "1,500"      → 1500
        """
        s = raw.strip()
        if not s or s in ("-", "—", "n/a", "na", "none"):
            return float("nan")

        # parenthesised negatives: (150) → -150
        if s.startswith("(") and s.endswith(")"):
            s = "-" + s[1:-1]

        # strip thousands-separator commas  ("1,500" → "1500")
        s = s.replace(",", "")

        # detect kN/m or kN unit suffix → multiply by 1e3
        factor = 1.0
        lower = s.lower()
        for suffix in ("kn/m", "kn·m/m", "kn·m", "kn"):
            if lower.endswith(suffix):
                s = s[: -len(suffix)].strip()
                factor = 1e3
                break

        try:
            return float(s) * factor
        except ValueError:
            raise ValueError(
                f"Cannot parse '{raw}' as a float for field '{field}' "
                f"(row {row_num}). Expected a number, optionally with kN/m units."
            )

    @classmethod
    def _normalise_headers(cls, raw_headers: list) -> dict:
        """
        Return a mapping {canonical_field: original_header} by matching
        raw CSV headers case-insensitively against known names and aliases.
        Unknown headers are kept as-is (they're ignored during parsing).
        """
        mapping = {}
        for h in raw_headers:
            key = h.strip().lower().replace(" ", "_")
            canonical = cls._ALIASES.get(key, None)
            if canonical is None:
                # Try direct case-insensitive match against known fields
                for fld in cls._FLOAT_FIELDS + ("name", "source", "description"):
                    if key == fld.lower():
                        canonical = fld
                        break
            if canonical:
                mapping[canonical] = h
        return mapping

    @classmethod
    def from_csv(cls, path: str) -> "LoadsDatabase":
        """
        Load from CSV.  Required columns: name, Nxx, Nyy, Nxy (case-insensitive).

        Handles common messy-data issues:
          - Column headers with wrong case or spaces (e.g. "NXX", "n xx")
          - Loads in kN/m instead of N/m (auto-detected from suffix)
          - Parenthesised negatives: (150) → -150
          - Thousands separators: 1,500 → 1500
          - Comment lines starting with '#'
          - BOM in UTF-8 files (utf-8-sig encoding)
          - Extra unknown columns (ignored)
        """
        cases = []
        try:
            fh = open(path, newline="", encoding="utf-8-sig")
        except UnicodeDecodeError:
            fh = open(path, newline="", encoding="latin-1")

        with fh as f:
            lines = [row for row in f if not row.lstrip().startswith("#")]

        reader = csv.DictReader(lines)
        hmap = cls._normalise_headers(reader.fieldnames or [])

        missing = [r for r in cls._REQUIRED if r not in hmap]
        if missing:
            raise ValueError(
                f"CSV '{path}' is missing required columns: {missing}. "
                f"Found headers: {reader.fieldnames}"
            )

        for row_num, row in enumerate(reader, start=2):
            name_raw = row.get(hmap["name"], "").strip()
            if not name_raw:
                name_raw = f"case_{row_num}"

            kwargs = {"name": name_raw}
            for fld in cls._FLOAT_FIELDS:
                col = hmap.get(fld)
                raw = row.get(col, "").strip() if col else ""
                val = cls._parse_float(raw, fld, row_num)
                if _np.isnan(val):
                    kwargs[fld] = 0.0 if fld != "eta" else float("nan")
                else:
                    kwargs[fld] = val

            kwargs["source"]      = row.get(hmap.get("source", ""), "").strip()
            kwargs["description"] = row.get(hmap.get("description", ""), "").strip()
            cases.append(LoadCase(**kwargs))

        return cls(cases)

    @classmethod
    def from_dict(cls, records: list) -> "LoadsDatabase":
        """
        Build a LoadsDatabase from a list of dicts (e.g. from JSON or a DataFrame).

        Each dict must contain 'name', 'Nxx', 'Nyy', 'Nxy' (case-insensitive).
        All other fields are optional.  String values with kN/m suffixes are
        parsed the same way as from_csv.

        Example
        -------
        >>> db = LoadsDatabase.from_dict([
        ...     {"name": "1g_cruise", "Nxx": "-120 kN/m", "Nyy": -40e3, "Nxy": 5e3},
        ...     {"Name": "3g_pull",   "NXX": -400e3,      "Nyy": -60e3, "Nxy": 12e3},
        ... ])
        """
        cases = []
        for row_num, record in enumerate(records, start=1):
            # Normalise keys
            norm = {}
            for k, v in record.items():
                key = k.strip().lower().replace(" ", "_")
                canonical = cls._ALIASES.get(key)
                if canonical is None:
                    for fld in cls._FLOAT_FIELDS + ("name", "source", "description"):
                        if key == fld.lower():
                            canonical = fld
                            break
                if canonical:
                    norm[canonical] = v

            missing = [r for r in cls._REQUIRED if r not in norm]
            if missing:
                raise ValueError(
                    f"Record {row_num} is missing required fields: {missing}. "
                    f"Got keys: {list(record.keys())}"
                )

            kwargs = {"name": str(norm["name"]).strip()}
            for fld in cls._FLOAT_FIELDS:
                raw = norm.get(fld, None)
                if raw is None:
                    kwargs[fld] = 0.0 if fld != "eta" else float("nan")
                elif isinstance(raw, str):
                    val = cls._parse_float(raw, fld, row_num)
                    kwargs[fld] = val if not _np.isnan(val) else (0.0 if fld != "eta" else float("nan"))
                else:
                    kwargs[fld] = float(raw)

            kwargs["source"]      = str(norm.get("source", "")).strip()
            kwargs["description"] = str(norm.get("description", "")).strip()
            cases.append(LoadCase(**kwargs))

        return cls(cases)

    def to_csv(self, path: str) -> None:
        """Write to CSV; round-trips via from_csv()."""
        fieldnames = [
            "name", "Nxx", "Nyy", "Nxy",
            "Mxx", "Myy", "Mxy",
            "source", "eta", "description",
        ]
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write("# LoadsDatabase — generated by composite_panel\n")
            f.write(f"# Units: N [N/m], M [N.m/m]\n")
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for c in self.cases:
                writer.writerow({
                    "name":        c.name,
                    "Nxx":         f"{c.Nxx:.2f}",
                    "Nyy":         f"{c.Nyy:.2f}",
                    "Nxy":         f"{c.Nxy:.2f}",
                    "Mxx":         f"{c.Mxx:.2f}",
                    "Myy":         f"{c.Myy:.2f}",
                    "Mxy":         f"{c.Mxy:.2f}",
                    "source":      c.source,
                    "eta":         f"{c.eta:.4f}" if not _np.isnan(c.eta) else "",
                    "description": c.description,
                })


    def filter_eta(self, eta: float, tol: float = 0.02) -> "LoadsDatabase":
        """Return cases within ±tol of spanwise station eta."""
        return LoadsDatabase(
            [c for c in self.cases if abs(c.eta - eta) <= tol]
        )

    def filter_source(self, source: str) -> "LoadsDatabase":
        """Return cases whose source string contains the given substring."""
        return LoadsDatabase(
            [c for c in self.cases if source.lower() in c.source.lower()]
        )

    def envelope_case(self, name: str = "envelope") -> LoadCase:
        """
        Conservative component-wise envelope (worst-case per component).

        Returns a single LoadCase with:
          Nxx = min(all Nxx)   — most compressive spanwise load
          Nyy = min(all Nyy)   — most compressive chordwise load
          Nxy = max(|all Nxy|) — largest magnitude shear

        WARNING: this is a conservative but potentially non-physical load
        combination — the worst Nxx and worst Nyy may not occur simultaneously.
        Use optimize_laminate_multicase() with ALL cases for the physically
        correct approach.

        Returns
        -------
        LoadCase  (source='envelope')
        """
        Nxx_vals = [c.Nxx for c in self.cases]
        Nyy_vals = [c.Nyy for c in self.cases]
        Nxy_vals = [c.Nxy for c in self.cases]
        Mxx_vals = [c.Mxx for c in self.cases]
        Myy_vals = [c.Myy for c in self.cases]
        Mxy_vals = [c.Mxy for c in self.cases]
        return LoadCase(
            name   = name,
            Nxx    = min(Nxx_vals),
            Nyy    = min(Nyy_vals),
            Nxy    = max(Nxy_vals, key=abs),
            Mxx    = min(Mxx_vals),
            Myy    = min(Myy_vals),
            Mxy    = max(Mxy_vals, key=abs),
            source = "envelope",
        )

    def summary(self) -> str:
        """Formatted table of all cases."""
        lines = [
            f"LoadsDatabase — {len(self.cases)} cases",
            f"  {'Name':<22} {'Nxx':>9} {'Nyy':>9} {'Nxy':>9}  {'Source':<14}  eta",
            f"  {'-'*22} {'-'*9} {'-'*9} {'-'*9}  {'-'*14}  ---",
        ]
        for c in self.cases:
            eta_str = f"{c.eta:.3f}" if not _np.isnan(c.eta) else "  —  "
            lines.append(
                f"  {c.name:<22} {c.Nxx/1e3:>+8.1f}k {c.Nyy/1e3:>+8.1f}k "
                f"{c.Nxy/1e3:>+8.1f}k  {c.source:<14}  {eta_str}"
            )
        # Show overall envelope
        env = self.envelope_case()
        lines += [
            f"  {'─'*22} {'─'*9} {'─'*9} {'─'*9}",
            f"  {'ENVELOPE':<22} {env.Nxx/1e3:>+8.1f}k {env.Nyy/1e3:>+8.1f}k "
            f"{env.Nxy/1e3:>+8.1f}k",
        ]
        return "\n".join(lines)

    def print_summary(self) -> None:
        print(self.summary())


if __name__ == "__main__":
    import sys as _sys
    _sys.stdout.reconfigure(encoding="utf-8")
    # Representative flight envelope for a supersonic wing skin at η=0.5
    records = [
        {"name": "M1.4_1g_mid",   "Nxx": -120e3, "Nyy":  -50e3, "Nxy": 18e3,
         "Mxx": 25.0, "source": "Ackeret", "eta": 0.5,
         "description": "M1.4 cruise 1g"},
        {"name": "M1.7_2.5g_mid", "Nxx": -280e3, "Nyy": -115e3, "Nxy": 42e3,
         "Mxx": 60.0, "source": "Ackeret", "eta": 0.5,
         "description": "M1.7 manoeuvre 2.5g"},
        {"name": "M2.0_1g_mid",   "Nxx": -160e3, "Nyy":  -65e3, "Nxy": 24e3,
         "Mxx": 32.0, "source": "Ackeret", "eta": 0.5,
         "description": "M2.0 cruise 1g"},
        {"name": "M2.0_2.5g_mid", "Nxx": -400e3, "Nyy": -160e3, "Nxy": 60e3,
         "Mxx": 80.0, "source": "Ackeret", "eta": 0.5,
         "description": "M2.0 manoeuvre 2.5g"},
    ]

    db = LoadsDatabase.from_dict(records)
    print(db.summary())
    print()

    env = db.envelope_case()
    print(f"Envelope:  Nxx={env.Nxx/1e3:.0f} kN/m,  "
          f"Nyy={env.Nyy/1e3:.0f} kN/m,  Nxy={env.Nxy/1e3:.0f} kN/m")
    print()

    # Scale by an additional 1.5 safety factor and show first case
    case = db[1].scaled(1.5)
    print(f"Scaled case: {case}")
