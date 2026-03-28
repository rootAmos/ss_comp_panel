"""
composite_panel.ply
-------------------
Single-ply material model.

Implements the reduced stiffness matrix Q (principal axes) and
the rotated stiffness matrix Q_bar (laminate axes) for a
unidirectional orthotropic ply under plane-stress assumption.

Reference:
    Kassapoglou, C. – Design and Analysis of Composite Structures
    (Wiley, 2013), Ch. 2
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PlyMaterial:
    """
    Unidirectional ply material properties.

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
        Material identifier
    """
    E1:   float
    E2:   float
    G12:  float
    nu12: float
    # Strengths (optional – needed for failure analysis)
    F1t:  Optional[float] = None
    F1c:  Optional[float] = None
    F2t:  Optional[float] = None
    F2c:  Optional[float] = None
    F12:  Optional[float] = None
    name: str = "unnamed"

    def __post_init__(self):
        self.nu21 = self.nu12 * self.E2 / self.E1   # minor Poisson

    @property
    def Q(self) -> np.ndarray:
        """
        3×3 reduced stiffness matrix in principal ply axes [Pa].

        [ Q11  Q12   0  ]
        [ Q12  Q22   0  ]
        [  0    0   Q66 ]
        """
        denom = 1.0 - self.nu12 * self.nu21
        Q11 = self.E1  / denom
        Q22 = self.E2  / denom
        Q12 = self.nu12 * self.E2 / denom
        Q66 = self.G12
        return np.array([[Q11, Q12,   0],
                         [Q12, Q22,   0],
                         [  0,   0, Q66]])


class Ply:
    """
    A single ply placed at a given orientation angle.

    Parameters
    ----------
    material : PlyMaterial
    thickness : float
        Ply thickness [m]
    angle_deg : float
        Fibre angle measured from the laminate x-axis [degrees]
    """

    def __init__(self, material: PlyMaterial, thickness: float, angle_deg: float):
        self.material  = material
        self.thickness = thickness
        self.angle_deg = angle_deg
        self._angle_rad = np.radians(angle_deg)

    # ------------------------------------------------------------------
    # Transformation matrix T  (Reuter convention)
    # ------------------------------------------------------------------
    @property
    def T(self) -> np.ndarray:
        """
        3×3 stress transformation matrix T such that
            σ_12 = T @ σ_xy
        """
        c = np.cos(self._angle_rad)
        s = np.sin(self._angle_rad)
        return np.array([[ c**2,  s**2,    2*c*s],
                         [ s**2,  c**2,   -2*c*s],
                         [-c*s,   c*s,  c**2-s**2]])

    @property
    def T_strain(self) -> np.ndarray:
        """
        3×3 strain transformation matrix (Reuter matrix applied).
        """
        c = np.cos(self._angle_rad)
        s = np.sin(self._angle_rad)
        return np.array([[ c**2,  s**2,    c*s],
                         [ s**2,  c**2,   -c*s],
                         [-2*c*s, 2*c*s,  c**2-s**2]])

    # ------------------------------------------------------------------
    # Rotated stiffness Q_bar
    # ------------------------------------------------------------------
    @property
    def Q_bar(self) -> np.ndarray:
        """
        3×3 reduced stiffness matrix rotated to laminate axes [Pa].

        Q_bar = T^-1 @ Q @ T_strain
        """
        T_inv = np.linalg.inv(self.T)
        return T_inv @ self.material.Q @ self.T_strain

    def __repr__(self) -> str:
        return (f"Ply(material='{self.material.name}', "
                f"t={self.thickness*1e3:.3f} mm, θ={self.angle_deg}°)")


# ---------------------------------------------------------------------------
# Built-in material presets (typical values – verify for your specific batch)
# ---------------------------------------------------------------------------

def IM7_8552() -> PlyMaterial:
    """
    Hexcel IM7/8552 Carbon/Epoxy – typical mid-plane properties.
    Suitable for supersonic airframe structural sizing (AM Duality™ context).

    Source: Hexcel HexPly 8552 datasheet + NIAR characterisation data.
    """
    return PlyMaterial(
        E1   = 171.4e9,   # Pa
        E2   =   9.08e9,  # Pa
        G12  =   5.29e9,  # Pa
        nu12 =   0.32,
        F1t  = 2326e6,    # Pa
        F1c  = 1200e6,    # Pa
        F2t  =    62e6,   # Pa
        F2c  =   200e6,   # Pa
        F12  =    90e6,   # Pa
        name = "IM7/8552",
    )


def T300_5208() -> PlyMaterial:
    """T300/5208 – classic aerospace benchmark material."""
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
