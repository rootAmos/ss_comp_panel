# External Load Sources

This repo is intended to size laminates from imported running loads.

The built-in `aero_loads.py` module is only a screening/demo path. For serious
use, aerodynamic pressures or panel running loads should come from an external
source such as CFD, wind-tunnel processing, or published benchmark datasets.

## Public benchmark examples

- ONERA M6 wing pressure-coefficient sections:
  NASA Glenn validation page
  https://www.grc.nasa.gov/WWW/wind/valid/m6wing/m6wing.html
- AGARD 445.6 wing benchmark overview:
  https://ntrl.ntis.gov/NTRL/dashboard/searchResults/titleDetail/N8827193.xhtml
- NASA discussion of AGARD 445.6 CFD/aeroelastic analyses:
  https://www.nas.nasa.gov/pubs/ams/2015/04-14-15.html
- NACA RM L58C07, supersonic pressure distributions on a 60 deg delta wing:
  https://ntrs.nasa.gov/citations/19660011611

## Expected import format

Convert your external aero result into panel running loads and import it through
`LoadsDatabase.from_csv()`.

CSV columns:

`name, Nxx, Nyy, Nxy, Mxx, Myy, Mxy, source, eta, description`

Conventions:

- `Nxx`, `Nyy`, `Nxy` are running forces in `N/m`
- `Mxx`, `Myy`, `Mxy` are running moments in `N*m/m`
- compression is negative
- `eta` is an optional spanwise station in `y / semi_span`

## Recommended workflow

1. Obtain aerodynamic pressure or panel load data from an external source.
2. Post-process it outside this repo into panel running loads.
3. Save those running loads to CSV using the schema above.
4. Import them with `LoadsDatabase.from_csv()`.
5. Run `optimize_laminate_multicase()` or `optimize_wing()` on the imported cases.

The repo does not currently attempt to provide a validated general-purpose
conversion from public pressure-coefficient benchmark data to full wing-skin
running loads.
