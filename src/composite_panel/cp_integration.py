"""
composite_panel.cp_integration
------------------------------
Cp-to-running-loads integration pipeline.

Takes pressure coefficient distributions (from CFD, wind tunnel, or
analytical models) and integrates them into panel running loads suitable
for structural sizing.

Pipeline:
    CpSection (raw Cp at one span station)
      -> section_coefficients()  -> Cn, Cm_le per section
    CpField   (sections across span for one flight condition)
      -> integrate_cp_field()    -> LoadsDatabase with panel running loads

Loaders:
    load_cp_csv()       -- from the repo's onera_m6_pressure_points.csv format
    load_cp_tecplot()   -- from Tecplot point-format (.tec) like ONERAb114.tec
"""

from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as _np

from .loads_db import LoadCase, LoadsDatabase

# numpy 2.x renamed trapz -> trapezoid
_trapz = getattr(_np, "trapezoid", None) or _np.trapz


# -----------------------------------------------------------------------
# Data containers
# -----------------------------------------------------------------------

@dataclass
class CpSection:
    """Pressure coefficient distribution at one spanwise section.

    *delta_Cp* = Cp_lower - Cp_upper (positive = net upward pressure = lift).
    Use :meth:`from_surfaces` if you have separate upper/lower arrays.
    """
    x_over_c: _np.ndarray   # chordwise stations [0..1], sorted ascending
    delta_Cp: _np.ndarray    # Cp_lower - Cp_upper at each station
    eta: float               # span fraction [0..1]

    @classmethod
    def from_surfaces(
        cls,
        x_upper: _np.ndarray, Cp_upper: _np.ndarray,
        x_lower: _np.ndarray, Cp_lower: _np.ndarray,
        eta: float,
    ) -> "CpSection":
        """Build from separate upper and lower surface Cp arrays.

        The two surfaces are interpolated onto a common x/c grid before
        computing delta_Cp = Cp_lower - Cp_upper.
        """
        x_upper = _np.asarray(x_upper, dtype=float)
        Cp_upper = _np.asarray(Cp_upper, dtype=float)
        x_lower = _np.asarray(x_lower, dtype=float)
        Cp_lower = _np.asarray(Cp_lower, dtype=float)

        # Sort both by x
        order_u = _np.argsort(x_upper)
        x_upper, Cp_upper = x_upper[order_u], Cp_upper[order_u]
        order_l = _np.argsort(x_lower)
        x_lower, Cp_lower = x_lower[order_l], Cp_lower[order_l]

        # Common grid: union of both, clipped to overlap
        x_min = max(x_upper[0], x_lower[0])
        x_max = min(x_upper[-1], x_lower[-1])
        x_common = _np.sort(_np.unique(_np.concatenate([x_upper, x_lower])))
        x_common = x_common[(x_common >= x_min) & (x_common <= x_max)]

        Cp_u_interp = _np.interp(x_common, x_upper, Cp_upper)
        Cp_l_interp = _np.interp(x_common, x_lower, Cp_lower)

        return cls(
            x_over_c=x_common,
            delta_Cp=Cp_l_interp - Cp_u_interp,
            eta=eta,
        )


@dataclass
class CpField:
    """Collection of :class:`CpSection` objects across the span for one
    flight condition.  Sections must be sorted by ascending *eta*."""
    sections: List[CpSection]
    mach: float
    alpha_deg: float
    q_inf: float                # dynamic pressure [Pa]
    source: str = ""
    description: str = ""

    def __post_init__(self):
        self.sections = sorted(self.sections, key=lambda s: s.eta)

    def __len__(self):
        return len(self.sections)

    @property
    def etas(self) -> _np.ndarray:
        return _np.array([s.eta for s in self.sections])


# -----------------------------------------------------------------------
# Section-level integration
# -----------------------------------------------------------------------

def section_coefficients(sec: CpSection) -> Tuple[float, float]:
    """Integrate Cp at one section to get aerodynamic coefficients.

    Returns
    -------
    Cn     : section normal-force coefficient  = integral delta_Cp d(x/c)
    Cm_le  : section pitching-moment coefficient about LE (positive nose-up)
             = -integral delta_Cp * (x/c) d(x/c)
    """
    if len(sec.x_over_c) < 2:
        return 0.0, 0.0
    Cn = float(_trapz(sec.delta_Cp, sec.x_over_c))
    Cm_le = float(-_trapz(sec.delta_Cp * sec.x_over_c, sec.x_over_c))
    return Cn, Cm_le


# -----------------------------------------------------------------------
# Spanwise integration  (Cp field -> panel running loads)
# -----------------------------------------------------------------------

def integrate_cp_field(
    cp_field: CpField,
    semi_span: float,
    chord_at_eta,                       # callable(eta) -> chord [m]
    box_height_at_eta,                  # callable(eta) -> structural height [m]
    box_chord_at_eta,                   # callable(eta) -> structural box width [m]
    sweep_deg_at_eta,                   # callable(eta) -> local LE sweep [deg]
    stringer_pitch: float = 0.180,      # [m]
    ea_chord_frac: float = 0.38,        # elastic axis as fraction of chord
    n_pts: int = 200,                   # interpolation resolution for spanwise integ.
) -> LoadsDatabase:
    """Integrate a :class:`CpField` into panel running loads.

    Steps:
      1. Integrate each section's Cp -> Cn, Cm_le
      2. Interpolate Cn(eta), Cm_le(eta) onto a fine spanwise grid
      3. Compute lift/span L'(eta), moment/span about EA
      4. Integrate spanwise for V(eta), M_bend(eta), T(eta)
      5. Convert to spar-aligned panel running loads at each *original*
         section station

    Parameters
    ----------
    cp_field          : CpField with sections and flight condition metadata
    semi_span         : half wingspan [m]
    chord_at_eta      : function returning local chord [m]
    box_height_at_eta : function returning structural box height [m]
    box_chord_at_eta  : function returning structural box width [m]
    sweep_deg_at_eta  : function returning local LE sweep [deg]
    stringer_pitch    : stringer spacing [m]
    ea_chord_frac     : elastic axis location as fraction of chord
    n_pts             : resolution for spanwise integration grid

    Returns
    -------
    LoadsDatabase with one LoadCase per section station.
    """
    q = cp_field.q_inf

    # Step 1: section coefficients at each measured station
    etas_data = cp_field.etas
    Cn_data = _np.zeros(len(cp_field))
    Cm_data = _np.zeros(len(cp_field))
    for i, sec in enumerate(cp_field.sections):
        Cn_data[i], Cm_data[i] = section_coefficients(sec)

    # Step 2: interpolate onto fine grid for spanwise integration
    eta_fine = _np.linspace(0.0, 1.0, n_pts)
    Cn_fine = _np.interp(eta_fine, etas_data, Cn_data,
                         left=Cn_data[0], right=0.0)
    Cm_fine = _np.interp(eta_fine, etas_data, Cm_data,
                         left=Cm_data[0], right=0.0)
    c_fine = _np.array([chord_at_eta(e) for e in eta_fine])

    y_fine = eta_fine * semi_span

    # Lift per unit span [N/m] and moment per unit span about EA [N]
    L_prime = Cn_fine * q * c_fine
    # Moment about EA per unit span:
    #   M_EA = q * c^2 * (Cm_le + Cn * ea_frac)
    M_ea_prime = q * c_fine ** 2 * (Cm_fine + Cn_fine * ea_chord_frac)

    # Step 3: spanwise integration for V, M_bend, T at fine grid
    V_fine = _np.zeros(n_pts)
    M_bend_fine = _np.zeros(n_pts)
    T_fine = _np.zeros(n_pts)
    for i in range(n_pts):
        sl = slice(i, None)
        y_sl = y_fine[sl]
        if len(y_sl) < 2:
            continue
        V_fine[i] = _trapz(L_prime[sl], y_sl)
        M_bend_fine[i] = _trapz(L_prime[sl] * (y_sl - y_fine[i]), y_sl)
        T_fine[i] = _trapz(M_ea_prime[sl], y_sl)

    # Step 4: panel running loads at each original section station
    cases: list[LoadCase] = []
    for sec in cp_field.sections:
        eta = sec.eta
        c_loc = chord_at_eta(eta)
        h_box = box_height_at_eta(eta)
        c_box = box_chord_at_eta(eta)
        sweep = _np.radians(sweep_deg_at_eta(eta))

        M_bend = float(_np.interp(eta, eta_fine, M_bend_fine))
        V_shear = float(_np.interp(eta, eta_fine, V_fine))
        T_torque = float(_np.interp(eta, eta_fine, T_fine))

        # Spar-aligned panel loads (same methodology as arrow_wing)
        cos_L = _np.cos(sweep)
        M_spar = M_bend * cos_L
        Nxx = -M_spar / max(h_box * c_box, 1e-6)

        # Pressure on panel between stringers
        Cn_sec, _ = section_coefficients(sec)
        delta_p_avg = Cn_sec * q  # average pressure difference
        Nyy = -delta_p_avg * stringer_pitch / 2.0

        # Shear from V and T
        Nxy_spar = abs(V_shear) / max(2.0 * h_box, 1e-6) * 0.25
        A_cell = h_box * c_box
        Nxy_torque = abs(T_torque) / max(2.0 * A_cell, 1e-6)
        Nxy = Nxy_spar + Nxy_torque

        Mxx = abs(delta_p_avg) * stringer_pitch ** 2 / 8.0

        mach_s = f"M{cp_field.mach}".replace(".", "p")
        name = f"Cp_{mach_s}_eta{int(round(eta * 100)):02d}"
        source = cp_field.source or "Cp-integration"

        cases.append(LoadCase(
            name=name, Nxx=float(Nxx), Nyy=float(Nyy), Nxy=float(Nxy),
            Mxx=float(Mxx), source=source, eta=eta,
            description=(f"Cp-integrated M{cp_field.mach} "
                         f"a={cp_field.alpha_deg}deg {cp_field.description}"),
        ))

    return LoadsDatabase(cases)


# -----------------------------------------------------------------------
# Loaders: CSV and Tecplot
# -----------------------------------------------------------------------

def load_cp_csv(
    path: str,
    mach: float = float("nan"),
    alpha_deg: float = float("nan"),
    q_inf: float = float("nan"),
    source: str = "",
) -> CpField:
    """Load Cp data from the repo's ``onera_m6_pressure_points.csv`` format.

    Expected columns: section, x_over_c, eta, cp, surface
    The *surface* column must be ``"upper"`` or ``"lower"``.
    """
    try:
        fh = open(path, newline="", encoding="utf-8-sig")
    except UnicodeDecodeError:
        fh = open(path, newline="", encoding="latin-1")

    with fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    # Group by (section, eta)
    sections_raw: dict[float, dict] = {}
    for row in rows:
        eta = float(row["eta"])
        xc = float(row["x_over_c"])
        cp = float(row["cp"])
        surf = row["surface"].strip().lower()

        if eta not in sections_raw:
            sections_raw[eta] = {"x_upper": [], "Cp_upper": [],
                                 "x_lower": [], "Cp_lower": []}
        if surf == "upper":
            sections_raw[eta]["x_upper"].append(xc)
            sections_raw[eta]["Cp_upper"].append(cp)
        elif surf == "lower":
            sections_raw[eta]["x_lower"].append(xc)
            sections_raw[eta]["Cp_lower"].append(cp)

    cp_sections = []
    for eta in sorted(sections_raw):
        d = sections_raw[eta]
        if not d["x_upper"] or not d["x_lower"]:
            continue
        sec = CpSection.from_surfaces(
            x_upper=_np.array(d["x_upper"]),
            Cp_upper=_np.array(d["Cp_upper"]),
            x_lower=_np.array(d["x_lower"]),
            Cp_lower=_np.array(d["Cp_lower"]),
            eta=eta,
        )
        cp_sections.append(sec)

    if not source:
        source = os.path.basename(path)

    return CpField(
        sections=cp_sections,
        mach=mach, alpha_deg=alpha_deg, q_inf=q_inf,
        source=source,
    )


def load_cp_tecplot(
    path: str,
    mach: float = float("nan"),
    alpha_deg: float = float("nan"),
    q_inf: float = float("nan"),
    source: str = "",
) -> CpField:
    """Load Cp data from Tecplot point-format files like ``ONERAb114.tec``.

    Expects VARIABLES including X/L (or X/C), Y/b (or ETA), Cp.
    Each ZONE is one spanwise section.  Points with negative Z/L are
    classified as lower surface, positive as upper.
    """
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()

    # Try to parse mach/alpha from TITLE line
    title_line = lines[0] if lines else ""
    if mach != mach:  # isnan
        m = re.search(r"(\d+\.\d+)\s+(\d+\.\d+)", title_line)
        if m:
            mach = float(m.group(1))
            alpha_deg = float(m.group(2))

    # Parse zones
    zones: list[dict] = []
    current_zone = None
    for line in lines:
        stripped = line.strip()
        if stripped.upper().startswith("ZONE"):
            if current_zone is not None:
                zones.append(current_zone)
            current_zone = {"points": []}
            continue
        if stripped.upper().startswith(("TITLE", "VARIABLES")):
            continue
        if current_zone is not None and stripped:
            parts = stripped.split()
            if len(parts) >= 5:
                try:
                    vals = [float(p) for p in parts]
                    current_zone["points"].append(vals)
                except ValueError:
                    pass
    if current_zone is not None:
        zones.append(current_zone)

    # Build CpSections from zones
    # Columns: NP, X/L, Y/b, Z/L, Cp
    cp_sections = []
    for zone in zones:
        pts = _np.array(zone["points"])
        if pts.size == 0 or pts.shape[1] < 5:
            continue
        xc_col = pts[:, 1]
        eta_col = pts[:, 2]
        zc_col = pts[:, 3]
        cp_col = pts[:, 4]

        eta = float(_np.median(eta_col))

        lower = zc_col <= 0
        upper = zc_col > 0

        if _np.sum(upper) < 2 or _np.sum(lower) < 2:
            continue

        sec = CpSection.from_surfaces(
            x_upper=xc_col[upper], Cp_upper=cp_col[upper],
            x_lower=xc_col[lower], Cp_lower=cp_col[lower],
            eta=eta,
        )
        cp_sections.append(sec)

    if not source:
        source = os.path.basename(path)

    return CpField(
        sections=cp_sections,
        mach=mach, alpha_deg=alpha_deg, q_inf=q_inf,
        source=source,
    )
