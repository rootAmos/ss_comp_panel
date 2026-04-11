"""
Ply material model and coordinate transforms for CLT.

Ref: Jones (1999) Ch. 2, Kassapoglou (2013) Ch. 2
"""

import aerosandbox.numpy as np
from dataclasses import dataclass, field
from typing import ClassVar, Optional


@dataclass
class PlyMaterial:
    """
    Orthotropic UD ply properties.  All values in Pa (moduli, strengths).
    Strengths stored as positive magnitudes; failure criteria flip sign internally.
    """
    E1:   float
    E2:   float
    G12:  float
    nu12: float
    F1t:  Optional[float] = None
    F1c:  Optional[float] = None
    F2t:  Optional[float] = None
    F2c:  Optional[float] = None
    F12:  Optional[float] = None
    name: str = "unnamed"

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
        self.nu21 = self.nu12 * self.E2 / self.E1   # Jones (1999), Eq. 2.59

    @classmethod
    def from_dict(cls, data: dict, units: str = "SI") -> "PlyMaterial":
        """
        Build from a dict.  units='SI' expects Pa; units='eng' expects GPa/MPa.
        Key aliases: e_fibre, e_transverse, g_shear, Xt/Xc/Yt/Yc/S12, etc.
        """
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

        if units == "eng":
            mod_scale = 1e9    # GPa -> Pa
            str_scale = 1e6    # MPa -> Pa
        elif units == "SI":
            mod_scale = 1.0
            str_scale = 1.0
        else:
            raise ValueError(f"units must be 'SI' or 'eng', got '{units}'")

        E1  = float(norm["E1"])  * mod_scale
        E2  = float(norm["E2"])  * mod_scale
        G12 = float(norm["G12"]) * mod_scale

        if E1 < 1e6 or E2 < 1e6:
            raise ValueError(
                f"Moduli look too small for Pa (E1={E1:.1f}, E2={E2:.1f}). "
                f"Pass units='eng' if values are in GPa/MPa."
            )

        return cls(
            E1   = E1,
            E2   = E2,
            G12  = G12,
            nu12 = float(norm["nu12"]),
            F1t  = float(norm["F1t"]) * str_scale if "F1t" in norm else None,
            F1c  = float(norm["F1c"]) * str_scale if "F1c" in norm else None,
            F2t  = float(norm["F2t"]) * str_scale if "F2t" in norm else None,
            F2c  = float(norm["F2c"]) * str_scale if "F2c" in norm else None,
            F12  = float(norm["F12"]) * str_scale if "F12" in norm else None,
            name = str(norm.get("name", "unnamed")),
        )

    @property
    def Q(self) -> np.ndarray:
        """Reduced stiffness [Pa].  Jones (1999), Eqs. 2.58-2.62."""
        denom = 1.0 - self.nu12 * self.nu21
        Q11 = self.E1  / denom
        Q22 = self.E2  / denom
        Q12 = self.nu12 * self.E2 / denom
        Q66 = self.G12
        return np.array([[Q11, Q12,   0],
                         [Q12, Q22,   0],
                         [  0,   0, Q66]])


class Ply:
    """Single UD ply at a given fibre angle.  Provides Q_bar, T, T_strain."""

    def __init__(self, material: PlyMaterial, thickness: float, angle_deg: float):
        self.material  = material
        self.thickness = thickness
        self.angle_deg = angle_deg
        self._angle_rad = np.radians(angle_deg)

    @property
    def T(self) -> np.ndarray:
        """Stress transformation matrix (laminate -> ply axes).  Jones (1999), Eq. 2.72."""
        c = np.cos(self._angle_rad)
        s = np.sin(self._angle_rad)
        return np.array([[ c**2,  s**2,    2*c*s],
                         [ s**2,  c**2,   -2*c*s],
                         [-c*s,   c*s,  c**2-s**2]])

    @property
    def T_strain(self) -> np.ndarray:
        """Strain transformation (laminate -> ply axes).  Jones (1999), Eq. 2.81."""
        c = np.cos(self._angle_rad)
        s = np.sin(self._angle_rad)
        return np.array([[ c**2,  s**2,    c*s],
                         [ s**2,  c**2,   -c*s],
                         [-2*c*s, 2*c*s,  c**2-s**2]])

    @property
    def Q_bar(self) -> np.ndarray:
        """Rotated stiffness in laminate axes.  Jones (1999), Eq. 2.84."""
        T_inv = np.linalg.inv(self.T)
        return T_inv @ self.material.Q @ self.T_strain

    def __repr__(self) -> str:
        return (f"Ply(material='{self.material.name}', "
                f"t={self.thickness*1e3:.3f} mm, theta={self.angle_deg}deg)")


# ---------------------------------------------------------------------------
# Built-in materials  --  nominal properties, not design allowables
# ---------------------------------------------------------------------------

def IM7_8552() -> PlyMaterial:
    """IM7/8552 UD carbon/epoxy.  Source: Hexcel (2020) product data sheet; CMH-17 Vol. 2."""
    return PlyMaterial(
        E1   = 171.4e9,
        E2   =   9.08e9,
        G12  =   5.29e9,
        nu12 =   0.32,
        F1t  = 2326e6,
        F1c  = 1200e6,
        F2t  =    62e6,
        F2c  =   200e6,
        F12  =    90e6,
        name = "IM7/8552",
    )


def T300_5208() -> PlyMaterial:
    """T300/5208 UD carbon/epoxy.  Source: Jones (1999), Table 2.1."""
    return PlyMaterial(
        E1   = 181e9,
        E2   =  10.3e9,
        G12  =   7.17e9,
        nu12 =   0.28,
        F1t  = 1500e6,
        F1c  = 1500e6,
        F2t  =    40e6,
        F2c  =   246e6,
        F12  =    68e6,
        name = "T300/5208",
    )


if __name__ == "__main__":
    import sys as _sys
    _sys.stdout.reconfigure(encoding="utf-8")
    mat = IM7_8552()
    print(f"Material: {mat.name}")
    print(f"  E1={mat.E1/1e9:.1f} GPa,  E2={mat.E2/1e9:.2f} GPa,  "
          f"G12={mat.G12/1e9:.2f} GPa,  nu12={mat.nu12}")
    print(f"  F1t={mat.F1t/1e6:.0f} MPa,  F1c={mat.F1c/1e6:.0f} MPa,  "
          f"F2t={mat.F2t/1e6:.0f} MPa,  F2c={mat.F2c/1e6:.0f} MPa,  "
          f"F12={mat.F12/1e6:.0f} MPa")
    print()

    cpt = 0.125e-3

    for angle in [0, 45, 90]:
        ply = Ply(mat, thickness=cpt, angle_deg=angle)
        print(f"Ply at theta={angle:3d}deg  Q_bar [GPa]:")
        for row in ply.Q_bar / 1e9:
            print("  " + "  ".join(f"{v:8.3f}" for v in row))
        print()
