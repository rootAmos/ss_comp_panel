"""
composite_panel.ply
-------------------
Single-ply material model for Classical Laminate Theory (CLT).

PHYSICS BACKGROUND
==================
A unidirectional (UD) composite ply is an orthotropic material — its stiffness
differs along three principal axes:
  - 1-axis : fibre direction          (high stiffness, ~E1 = 171 GPa for IM7)
  - 2-axis : transverse to fibre      (matrix-dominated, ~E2 = 9 GPa)
  - 3-axis : through-thickness        (ignored under plane-stress assumption)

Under the plane-stress assumption (valid for thin panels, σ3 = τ13 = τ23 = 0),
the in-plane constitutive relation reduces to:

    [σ1 ]   [Q11  Q12   0 ] [ε1 ]
    [σ2 ] = [Q12  Q22   0 ] [ε2 ]
    [τ12]   [ 0    0  Q66] [γ12]

where Qij are the REDUCED stiffness coefficients (derived from E1, E2, G12, ν12).

When a ply is oriented at angle θ to the laminate x-axis, the Q matrix must be
rotated into laminate coordinates. This is done via the transformation matrices
T and T_strain, giving the ROTATED stiffness matrix Q_bar.  It is Q_bar — not Q —
that is integrated through the thickness to build the laminate ABD matrix.

DESIGN CONTEXT
==============
For a supersonic airframe skin (Mach 1.4-3.0), IM7/8552 is the go-to material:
  - High specific stiffness  → minimises aeroelastic deflection at speed
  - High F1t/F1c ratio       → efficient under the compression-dominated
                                wing bending loads typical of upper skin panels
  - Verified NIAR allowables → accepted for FAA/EASA structural substantiation

Reference:
    Kassapoglou, C. – Design and Analysis of Composite Structures
    (Wiley, 2013), Ch. 2

    Jones, R.M. – Mechanics of Composite Materials (Taylor & Francis, 1999)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import ClassVar, Optional


@dataclass
class PlyMaterial:
    """
    Orthotropic material properties for a single unidirectional ply.

    All elastic constants follow the standard composite notation:
      E1, E2  : Young's moduli in fibre and transverse directions [Pa]
      G12     : In-plane shear modulus [Pa]
      nu12    : Major Poisson's ratio (fibre-loading → transverse contraction) [-]
      nu21    : Minor Poisson's ratio — derived via Maxwell reciprocity:
                    nu21 = nu12 * E2 / E1                (computed automatically)

    Strength allowables (needed for failure analysis):
      F1t, F1c : Longitudinal tension / compression strength [Pa]
      F2t, F2c : Transverse  tension / compression strength [Pa]
      F12      : In-plane shear strength [Pa]

    Note on sign convention for F1c, F2c:
        Store as positive magnitudes.  The failure criteria flip the sign
        internally when the relevant stress component is compressive.

    Parameters
    ----------
    E1 : float
        Longitudinal (fibre-direction) Young's modulus [Pa]
    E2 : float
        Transverse Young's modulus [Pa]
    G12 : float
        In-plane shear modulus [Pa]
    nu12 : float
        Major Poisson's ratio [-]
    F1t : float, optional
        Longitudinal tensile strength [Pa]
    F1c : float, optional
        Longitudinal compressive strength [Pa]  (positive value)
    F2t : float, optional
        Transverse tensile strength [Pa]
    F2c : float, optional
        Transverse compressive strength [Pa]    (positive value)
    F12 : float, optional
        In-plane shear strength [Pa]
    name : str, optional
        Material identifier string
    """
    E1:   float
    E2:   float
    G12:  float
    nu12: float
    # Strengths are optional — CLT stiffness analysis does not need them,
    # but any failure check will raise an error if they are None.
    F1t:  Optional[float] = None
    F1c:  Optional[float] = None
    F2t:  Optional[float] = None
    F2c:  Optional[float] = None
    F12:  Optional[float] = None
    name: str = "unnamed"

    # Key aliases for from_dict — maps lowercase variant → canonical attribute.
    _ALIASES: ClassVar[dict] = {
        "e_1": "E1", "e1": "E1", "e_fibre": "E1", "e_fiber": "E1",
        "e_2": "E2", "e2": "E2", "e_transverse": "E2", "e_matrix": "E2",
        "g_12": "G12", "g12": "G12", "g_shear": "G12",
        "nu_12": "nu12", "nu12": "nu12", "v12": "nu12", "poisson": "nu12",
        "f_1t": "F1t", "f1t": "F1t", "xt": "F1t",
        "f_1c": "F1c", "f1c": "F1c", "xc": "F1c",
        "f_2t": "F2t", "f2t": "F2t", "yt": "F2t",
        "f_2c": "F2c", "f2c": "F2c", "yc": "F2c",
        "f_12": "F12", "f12": "F12", "s12": "F12", "s": "F12",
    }

    def __post_init__(self):
        # Maxwell reciprocity: nu21/E2 = nu12/E1  →  nu21 = nu12 * E2 / E1
        # This is NOT a free parameter — it is fully determined by nu12 and the
        # elastic moduli.  Violating this relation would break energy conservation.
        self.nu21 = self.nu12 * self.E2 / self.E1

    @classmethod
    def from_dict(cls, data: dict) -> "PlyMaterial":
        """
        Build a PlyMaterial from a dict, handling mixed units and key aliases.

        Units are auto-detected per field:
          - Moduli (E1, E2, G12): values <= 1000 assumed GPa → converted to Pa
          - Strengths (F1t etc.): values <= 10000 assumed MPa → converted to Pa
          - nu12: dimensionless, no conversion

        Key aliases accepted (case-insensitive):
          E1 / e_1 / e_fibre / e_fiber
          E2 / e_2 / e_transverse
          G12 / g_12 / g_shear
          nu12 / nu_12 / v12 / poisson
          F1t / f_1t / Xt,  F1c / f_1c / Xc
          F2t / f_2t / Yt,  F2c / f_2c / Yc
          F12 / f_12 / S12 / S

        Example
        -------
        >>> mat = PlyMaterial.from_dict({
        ...     "E1": 171.4,   # GPa — auto-converted
        ...     "E2": 9.08,    # GPa
        ...     "G12": 5.29,   # GPa
        ...     "nu12": 0.32,
        ...     "F1t": 2326,   # MPa — auto-converted
        ...     "F1c": 1200, "F2t": 62, "F2c": 200, "F12": 110,
        ...     "name": "IM7/8552",
        ... })
        """
        # Normalise keys
        norm = {}
        for k, v in data.items():
            key = k.strip().lower().replace(" ", "_")
            canonical = cls._ALIASES.get(key)
            if canonical is None and key in ("e1", "e2", "g12", "nu12",
                                              "f1t", "f1c", "f2t", "f2c", "f12"):
                canonical = key.upper() if key.startswith(("e", "g", "f")) else key
            if canonical:
                norm[canonical] = v
            elif key == "name":
                norm["name"] = v

        required = ("E1", "E2", "G12", "nu12")
        missing = [r for r in required if r not in norm]
        if missing:
            raise ValueError(
                f"PlyMaterial.from_dict: missing required fields {missing}. "
                f"Got: {list(data.keys())}"
            )

        def _to_pa_modulus(v):
            v = float(v)
            return v * 1e9 if v <= 1000 else v   # GPa → Pa if value ≤ 1000

        def _to_pa_strength(v):
            v = float(v)
            return v * 1e6 if v <= 10000 else v  # MPa → Pa if value ≤ 10000

        return cls(
            E1   = _to_pa_modulus(norm["E1"]),
            E2   = _to_pa_modulus(norm["E2"]),
            G12  = _to_pa_modulus(norm["G12"]),
            nu12 = float(norm["nu12"]),
            F1t  = _to_pa_strength(norm["F1t"]) if "F1t" in norm else None,
            F1c  = _to_pa_strength(norm["F1c"]) if "F1c" in norm else None,
            F2t  = _to_pa_strength(norm["F2t"]) if "F2t" in norm else None,
            F2c  = _to_pa_strength(norm["F2c"]) if "F2c" in norm else None,
            F12  = _to_pa_strength(norm["F12"]) if "F12" in norm else None,
            name = str(norm.get("name", "unnamed")),
        )

    @property
    def Q(self) -> np.ndarray:
        """
        3×3 reduced stiffness matrix in principal ply axes [Pa].

        Derived by inverting the plane-stress compliance matrix and eliminating
        the through-thickness terms.  The denominator (1 - ν12·ν21) appears
        because Poisson coupling between the two in-plane directions partially
        stiffens the response — analogous to the biaxial constraint factor in
        isotropic materials.

        Matrix structure:
            [ Q11  Q12   0  ]       Q11 = E1  / (1 - ν12·ν21)
            [ Q12  Q22   0  ]       Q22 = E2  / (1 - ν12·ν21)
            [  0    0   Q66 ]       Q12 = ν12·E2 / (1 - ν12·ν21)
                                    Q66 = G12   (shear decouples under plane stress)

        Returns
        -------
        np.ndarray, shape (3, 3)
        """
        denom = 1.0 - self.nu12 * self.nu21   # Poisson coupling denominator
        Q11 = self.E1  / denom
        Q22 = self.E2  / denom
        Q12 = self.nu12 * self.E2 / denom     # = nu21 * E1 / denom (symmetric)
        Q66 = self.G12
        return np.array([[Q11, Q12,   0],
                         [Q12, Q22,   0],
                         [  0,   0, Q66]])


class Ply:
    """
    A single UD ply at a specified fibre orientation angle.

    The ply angle θ is measured from the global laminate x-axis to the
    fibre direction, positive counterclockwise.  For a wing-skin laminate,
    x is typically the spanwise direction:

        θ =   0° → fibres run spanwise      (maximum bending stiffness contribution)
        θ =  90° → fibres run chordwise     (max transverse stiffness / hoop loads)
        θ = ±45° → fibres at 45°            (shear-dominant; torsion box loads)

    The orientation determines Q_bar — the stiffness seen by the laminate —
    which drives all the ABD integrals and, ultimately, the stress state.

    Parameters
    ----------
    material : PlyMaterial
        Orthotropic material data for this ply.
    thickness : float
        Cured ply thickness [m].  Typical prepreg CPT ≈ 0.125 mm.
    angle_deg : float
        Fibre angle measured from laminate x-axis [degrees].
    """

    def __init__(self, material: PlyMaterial, thickness: float, angle_deg: float):
        self.material  = material
        self.thickness = thickness
        self.angle_deg = angle_deg
        self._angle_rad = np.radians(angle_deg)   # cache for repeated use

    # ------------------------------------------------------------------
    # Transformation matrices
    # ------------------------------------------------------------------

    @property
    def T(self) -> np.ndarray:
        """
        3×3 stress transformation matrix (laminate → ply axes).

        Rotates a stress vector from the laminate (x-y) coordinate system into
        the ply principal (1-2) axes:

            [σ1 ]           [σx ]
            [σ2 ] = T  @    [σy ]
            [τ12]           [τxy]

        Derived from Mohr's circle extended to 3 components.  The factor of 2
        on the shear-coupling terms arises because engineering shear strain
        (γ12 = 2·ε12) is used rather than tensorial shear strain.

        This is the Reuter convention — consistent with the T_strain definition
        below so that T_strain = Reuter @ T @ Reuter^-1.

        Returns
        -------
        np.ndarray, shape (3, 3)
        """
        c = np.cos(self._angle_rad)
        s = np.sin(self._angle_rad)
        return np.array([[ c**2,  s**2,    2*c*s],
                         [ s**2,  c**2,   -2*c*s],
                         [-c*s,   c*s,  c**2-s**2]])

    @property
    def T_strain(self) -> np.ndarray:
        """
        3×3 strain transformation matrix (laminate → ply axes).

        Transforms the engineering strain vector [εx, εy, γxy] into principal
        axes [ε1, ε2, γ12].  The off-diagonal factors differ from T by a factor
        of 1/2 on the shear coupling terms — this corrects for the engineering
        vs tensorial shear strain convention.

        Relationship: T_strain = R @ T @ R^-1  where R = diag(1, 1, 2) is the
        Reuter matrix that maps engineering → tensorial shear strain.

        Returns
        -------
        np.ndarray, shape (3, 3)
        """
        c = np.cos(self._angle_rad)
        s = np.sin(self._angle_rad)
        return np.array([[ c**2,  s**2,    c*s],
                         [ s**2,  c**2,   -c*s],
                         [-2*c*s, 2*c*s,  c**2-s**2]])

    # ------------------------------------------------------------------
    # Rotated stiffness matrix
    # ------------------------------------------------------------------

    @property
    def Q_bar(self) -> np.ndarray:
        """
        3×3 reduced stiffness matrix rotated to laminate axes [Pa].

        This is the key quantity used in CLT integration.  It represents the
        apparent stiffness of this ply *as seen from the global laminate frame*.

        Derivation:
            σ_12  = Q  @ ε_12                  (constitutive in ply axes)
            σ_12  = T  @ σ_xy                  (rotate stress to ply axes)
            ε_12  = T_strain @ ε_xy             (rotate strain to ply axes)
            →  σ_xy = T^-1 @ Q @ T_strain @ ε_xy
            →  Q_bar = T^-1 @ Q @ T_strain

        Physical interpretation:
          A 0° ply has Q_bar = Q (no rotation needed).
          A 45° ply spreads stiffness more evenly across all components,
          contributing significantly to Q_bar_66 (in-plane shear stiffness).
          A 90° ply swaps Q11↔Q22, making the transverse direction stiff
          and the fibre direction compliant relative to global axes.

        Returns
        -------
        np.ndarray, shape (3, 3)
        """
        T_inv = np.linalg.inv(self.T)
        return T_inv @ self.material.Q @ self.T_strain

    def __repr__(self) -> str:
        return (f"Ply(material='{self.material.name}', "
                f"t={self.thickness*1e3:.3f} mm, θ={self.angle_deg}°)")


# ---------------------------------------------------------------------------
# Built-in material presets
# ---------------------------------------------------------------------------
# These are mid-plane typical properties from published datasheets.
# For flight-critical structural sizing, use design allowables (B-basis or
# A-basis values) from a certified test programme, not these nominal values.
# ---------------------------------------------------------------------------

def IM7_8552() -> PlyMaterial:
    """
    Hexcel IM7/8552 unidirectional carbon/epoxy — nominal mid-plane properties.

    IM7/8552 is the workhorse of high-performance aerospace composite structures:
      - IM7 intermediate-modulus carbon fibre: high strength-to-weight, good
        strain-to-failure for damage tolerance
      - 8552 toughened epoxy matrix: excellent compression-after-impact (CAI)
        and hot/wet retention — critical for supersonic skin panels that see
        elevated temperatures from aerodynamic heating

    Property sources:
      - Hexcel HexPly 8552 product data sheet (nominal values)
      - NIAR AGATE/NCAMP characterisation data (statistical allowables basis)

    Typical usage context:
      Upper/lower wing skin panels, fuselage frames, control surfaces on
      high-performance aircraft where weight, stiffness, and fatigue life all
      drive the design simultaneously.

    Returns
    -------
    PlyMaterial
    """
    return PlyMaterial(
        E1   = 171.4e9,   # Pa  — fibre-dominated; very consistent lot-to-lot
        E2   =   9.08e9,  # Pa  — matrix-dominated; sensitive to cure cycle
        G12  =   5.29e9,  # Pa  — shear modulus
        nu12 =   0.32,    # [-] — typical for CFRP; drives Poisson coupling in CLT
        F1t  = 2326e6,    # Pa  — fibre tensile: carbon fibre controls
        F1c  = 1200e6,    # Pa  — fibre compression: buckling/kinking driven,
                          #        significantly lower than F1t — watch for
                          #        upper-skin compression designs
        F2t  =    62e6,   # Pa  — transverse tensile: matrix cracking limit;
                          #        low — avoid high σ2 tension in layup design
        F2c  =   200e6,   # Pa  — transverse compression: higher than F2t
                          #        (matrix confined laterally)
        F12  =    90e6,   # Pa  — shear strength; ±45° plies govern this
        name = "IM7/8552",
    )


def T300_5208() -> PlyMaterial:
    """
    T300/5208 unidirectional carbon/epoxy — classic aerospace benchmark material.

    T300/5208 is the reference system used in most composite textbook examples
    (Tsai, Jones, Kassapoglou) and early composite certification programmes.
    Lower performance than IM7/8552 — kept here as a validation reference and
    for legacy analyses.

    Returns
    -------
    PlyMaterial
    """
    return PlyMaterial(
        E1   = 181e9,     # Pa  — slightly stiffer fibre than IM7 in literature values
        E2   =  10.3e9,   # Pa
        G12  =   7.17e9,  # Pa
        nu12 =   0.28,    # [-]
        F1t  = 1500e6,    # Pa
        F1c  = 1500e6,    # Pa  — textbook often quotes equal t/c (conservative)
        F2t  =    40e6,   # Pa
        F2c  =   246e6,   # Pa
        F12  =    68e6,   # Pa
        name = "T300/5208",
    )
