# composite_panel

Classical Laminate Theory toolkit for sizing composite skin panels on supersonic and hypersonic airframes. Covers aerodynamic loads, CLT stiffness and stress recovery, failure criteria, panel buckling, thermal loads, flight mechanics trim, static aeroelastic correction, a CSV-backed loads database, and a gradient-based minimum-mass laminate optimizer.

---

## Structure

```
src/composite_panel/
├── ply.py           — PlyMaterial, Ply, IM7_8552, T300_5208
├── laminate.py      — ABD matrix, CLT response, effective moduli, alpha_lam
├── failure.py       — Tsai-Wu, Tsai-Hill, Hashin, max-stress, max-strain
├── buckling.py      — panel buckling RF under combined Nxx/Nyy/Nxy
├── thermal.py       — CTE transforms, thermal resultants, aero heating
├── aero_loads.py    — Ackeret/oblique-shock/Prandtl-Glauert pressure → running loads
├── loads_db.py      — LoadCase, LoadsDatabase (from_csv, to_csv, filter, envelope)
├── trim.py          — flight mechanics trim (α, CL, q∞) across Mach range
├── aeroelastic.py   — static aeroelastic correction (Euler-Bernoulli + washout)
└── optimizer.py     — minimum-mass NLP via IPOPT/CasADi (single + multicase)

scripts/
├── demo_supersonic_panel.py   — CLT + failure + stacking trade for a single panel
├── demo_multicase_sizing.py   — single-case vs multi-case optimizer comparison
├── demo_hypersonic_wing.py    — spanwise skin sizing from Mach 0.8 to Mach 5
└── validate_model.py          — regression checks against analytical benchmarks

tests/
├── test_basics.py
├── test_clt.py
├── test_failure.py
├── test_buckling.py
└── test_input_parsing.py

docs/
└── optimizer.md     — NLP formulation, CasADi implementation notes, references

notebooks/
├── composite_panel_tutorial.ipynb
├── tutorial_supersonic_panel.ipynb
├── tutorial_multicase_sizing.ipynb
└── tutorial_hypersonic_wing.ipynb
```

---

## Installation

```bash
pip install -e ".[dev]"
```

Core dependencies (`numpy`, `matplotlib`, `aerosandbox`) are declared in `pyproject.toml`. `aerosandbox` pulls in CasADi, which is required for the optimizer.

---

## Testing and validation

### Unit tests

```bash
pytest tests/
```

67 tests across CLT, failure criteria (Tsai-Wu, Hashin, max-stress), buckling (including bend-twist coupling detection), input parsing, and basic integration checks.

### Analytical validation

```bash
python scripts/validate_model.py
```

`validate_model.py` checks the implementation against known analytical results across 7 blocks:

| Block | What is checked | Reference |
|---|---|---|
| 1 | IM7/8552 elastic constants and strengths vs CMH-17 B-basis | CMH-17-1F Vol. 2 Ch. 4 |
| 2 | CLT limit cases — A11 exact formula, B=0 for symmetric layups, quasi-isotropic symmetry | Jones (1999) |
| 3 | Tsai-Wu RF = 1.0 at each uniaxial failure boundary; linear scaling with load | Tsai & Wu (1971) |
| 4 | Nxx_cr matches Timoshenko closed form for [0]8; cubic h³ scaling | Timoshenko & Gere (1961) |
| 5 | Ackeret ΔCp within 10% of oblique shock at M=1.5; Prandtl-Glauert at M=0.6 | Ackeret (1925) |
| 6 | Optimizer KKT: RF at optimum equals rf_min (active constraint) | — |
| 7 | Physics monotonicity: mass increases with Mach and load factor; root thicker than tip | — |

Output is a colour-coded PASS/FAIL summary with tolerances and deviation from reference values.

---

## Demos

```bash
python scripts/demo_supersonic_panel.py
python scripts/demo_multicase_sizing.py
python scripts/demo_hypersonic_wing.py
```

Outputs are written to `outputs/`.

---

## Key modules

### `laminate.py`

```python
from composite_panel import Ply, Laminate, IM7_8552

mat   = IM7_8552()
plies = [Ply(mat, 0.125e-3, θ) for θ in [0, 45, -45, 90, 90, -45, 45, 0]]
lam   = Laminate(plies)

print(lam.summary())          # A, D matrices + Ex, Ey, Gxy
res = lam.response(N=[-500e3, -50e3, 20e3])   # midplane strains, ply stresses
```

### `failure.py`

```python
from composite_panel import check_laminate

fails = check_laminate(res, plies, criterion='tsai_wu')   # or 'hashin', 'max_stress', ...
gov   = min(fails, key=lambda r: r.rf)
print(f"Governing RF = {gov.rf:.3f}  (ply {gov.ply_index}, θ={gov.angle_deg}°)")
```

### `loads_db.py`

```python
from composite_panel import LoadCase, LoadsDatabase

db = LoadsDatabase.from_csv("scripts/flight_envelope.csv")
station_cases = db.filter_eta(0.45)   # all cases at 45% semi-span
worst         = db.envelope_case()    # single worst-case envelope load
```

### `optimizer.py`

Single load case:

```python
from composite_panel import optimize_laminate, detect_balance_pairs, IM7_8552
import numpy as np

mat    = IM7_8552()
angles = [0.0, 45.0, -45.0, 90.0]
pairs  = detect_balance_pairs(angles)

result = optimize_laminate(
    N_loads         = np.array([-500e3, -50e3, 20e3]),
    M_loads         = np.zeros(3),
    mat             = mat,
    angles_half_deg = angles,
    rf_min          = 1.5,
    balance_pairs   = pairs,
)
print(result.summary())
```

All load cases simultaneously:

```python
from composite_panel import optimize_laminate_multicase, LoadsDatabase

db     = LoadsDatabase.from_csv("scripts/flight_envelope.csv")
result = optimize_laminate_multicase(
    load_cases      = db.filter_eta(0.45),
    mat             = mat,
    angles_half_deg = angles,
    rf_min          = 1.5,
    balance_pairs   = pairs,
)
print(f"Areal density : {result.areal_density:.3f} kg/m²")
print(f"Min RF        : {result.min_tsai_wu_rf:.3f}")
```

Spanwise wing sizing:

```python
from composite_panel import optimize_wing, WingGeometry

wing = WingGeometry(semi_span=4.5, root_chord=4.0, taper_ratio=0.25,
                    sweep_le_deg=50.0, t_over_c=0.04, mtow_n=120_000.0)

r = optimize_wing(wing=wing, mach=1.7, altitude_m=20_000, alpha_deg=3.0,
                  mat=mat, angles_half_deg=angles, n_load=2.5,
                  rf_min=1.5, balance_pairs=pairs)

print(f"Upper-skin mass (semi-span): {r.total_skin_mass:.1f} kg")
```

### `thermal.py`

```python
from composite_panel import IM7_8552_thermal, thermal_state_from_flight

pt    = IM7_8552_thermal()                        # α₁, α₂, T_cure for IM7/8552
state = thermal_state_from_flight(mach=4.0, altitude_m=25_000)

# Effective laminate CTE
alpha = lam.alpha_lam([pt] * len(plies))          # [αx, αy, αxy] in 1/K
```

---

## References

| Topic | Reference |
|---|---|
| CLT | Jones — *Mechanics of Composite Materials* (1999) |
| Tsai-Wu | Tsai & Wu — *J. Composite Materials* 5(1), 1971 |
| Hashin | Hashin — *J. Applied Mechanics* 47(2), 1980 |
| Buckling | Timoshenko & Gere — *Theory of Elastic Stability* (1961) Ch. 9 |
| Buckling (shear) | ESDU 02.03.11 |
| Ackeret pressure | Ackeret — *ZAMM* 5(1), 1925 |
| Optimizer | Kassapoglou — *Design and Analysis of Composite Structures* (2013) |
