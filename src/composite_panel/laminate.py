"""
composite_panel.laminate
------------------------
Classical Laminate Theory (CLT) implementation.

Given an ordered stack of Ply objects (bottom → top), computes:
  - A, B, D submatrices  (extensional, coupling, bending stiffness)
  - ABD matrix and its inverse
  - Midplane strains and curvatures for applied loads/moments
  - Ply-level stresses and strains (both laminate and principal axes)

Reference:
    Kassapoglou, C. – Design and Analysis of Composite Structures
    (Wiley, 2013), Ch. 3–4
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional

from .ply import Ply


class Laminate:
    """
    Composite laminate defined by an ordered list of plies.

    Parameters
    ----------
    plies : list of Ply
        Ordered from BOTTOM (z = -h/2) to TOP (z = +h/2).

    Notes
    -----
    z = 0 is the laminate mid-plane.
    """

    def __init__(self, plies: List[Ply]):
        self.plies = plies
        self._build_z_coords()

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    def _build_z_coords(self):
        """Compute ply interface z-coordinates (bottom → top)."""
        thicknesses = [p.thickness for p in self.plies]
        total = sum(thicknesses)
        self._h = total                          # total laminate thickness

        # z_k : bottom face of ply k  (z_0 = -h/2)
        self._z = np.zeros(len(self.plies) + 1)
        self._z[0] = -total / 2.0
        for k, t in enumerate(thicknesses):
            self._z[k + 1] = self._z[k] + t

    @property
    def thickness(self) -> float:
        """Total laminate thickness [m]."""
        return self._h

    @property
    def z_interfaces(self) -> np.ndarray:
        """z-coordinates of ply interfaces [m], length = n_plies + 1."""
        return self._z

    # ------------------------------------------------------------------
    # ABD matrix
    # ------------------------------------------------------------------

    def _compute_ABD(self):
        A = np.zeros((3, 3))
        B = np.zeros((3, 3))
        D = np.zeros((3, 3))
        for k, ply in enumerate(self.plies):
            Qb  = ply.Q_bar
            z0  = self._z[k]
            z1  = self._z[k + 1]
            A  += Qb * (z1 - z0)
            B  += Qb * (z1**2 - z0**2) / 2.0
            D  += Qb * (z1**3 - z0**3) / 3.0
        self._A = A
        self._B = B
        self._D = D
        self._ABD = np.block([[A, B],
                               [B, D]])
        self._abd = np.linalg.inv(self._ABD)   # compliance

    @property
    def A(self) -> np.ndarray:
        """3×3 extensional stiffness matrix [N/m]."""
        if not hasattr(self, '_A'):
            self._compute_ABD()
        return self._A

    @property
    def B(self) -> np.ndarray:
        """3×3 coupling stiffness matrix [N]."""
        if not hasattr(self, '_B'):
            self._compute_ABD()
        return self._B

    @property
    def D(self) -> np.ndarray:
        """3×3 bending stiffness matrix [N·m]."""
        if not hasattr(self, '_D'):
            self._compute_ABD()
        return self._D

    @property
    def ABD(self) -> np.ndarray:
        """6×6 full stiffness matrix."""
        if not hasattr(self, '_ABD'):
            self._compute_ABD()
        return self._ABD

    @property
    def abd(self) -> np.ndarray:
        """6×6 compliance matrix (inverse of ABD)."""
        if not hasattr(self, '_abd'):
            self._compute_ABD()
        return self._abd

    # ------------------------------------------------------------------
    # Effective engineering constants (symmetric laminates only)
    # ------------------------------------------------------------------

    @property
    def Ex(self) -> float:
        """Effective longitudinal modulus [Pa] (symmetric laminates)."""
        return 1.0 / (self.abd[0, 0] * self._h)

    @property
    def Ey(self) -> float:
        """Effective transverse modulus [Pa] (symmetric laminates)."""
        return 1.0 / (self.abd[1, 1] * self._h)

    @property
    def Gxy(self) -> float:
        """Effective shear modulus [Pa] (symmetric laminates)."""
        return 1.0 / (self.abd[2, 2] * self._h)

    # ------------------------------------------------------------------
    # Load response
    # ------------------------------------------------------------------

    def response(self,
                 N: Optional[np.ndarray] = None,
                 M: Optional[np.ndarray] = None) -> dict:
        """
        Compute midplane strains, curvatures, and ply stresses.

        Parameters
        ----------
        N : array-like, shape (3,)  [N/m]
            Running loads  [Nxx, Nyy, Nxy].  Default zeros.
        M : array-like, shape (3,)  [N·m/m]
            Running moments [Mxx, Myy, Mxy].  Default zeros.

        Returns
        -------
        dict with keys:
            'eps0'          : midplane strains   (3,)
            'kappa'         : curvatures         (3,)
            'ply_strain_xy' : list of (3,) arrays – strains  in laminate axes per ply (at mid-ply z)
            'ply_stress_xy' : list of (3,) arrays – stresses in laminate axes per ply
            'ply_stress_12' : list of (3,) arrays – stresses in principal ply axes
            'ply_strain_12' : list of (3,) arrays – strains  in principal ply axes
        """
        N = np.zeros(3) if N is None else np.asarray(N, dtype=float)
        M = np.zeros(3) if M is None else np.asarray(M, dtype=float)

        load_vec = np.concatenate([N, M])
        deform   = self.abd @ load_vec
        eps0     = deform[:3]
        kappa    = deform[3:]

        ply_strain_xy, ply_stress_xy = [], []
        ply_strain_12, ply_stress_12 = [], []

        for k, ply in enumerate(self.plies):
            z_mid = (self._z[k] + self._z[k + 1]) / 2.0

            # Strain in laminate axes at mid-ply z
            eps_xy = eps0 + z_mid * kappa

            # Stress in laminate axes
            sig_xy = ply.Q_bar @ eps_xy

            # Rotate to principal ply axes
            T      = ply.T
            T_s    = ply.T_strain
            sig_12 = T @ sig_xy
            eps_12 = T_s @ eps_xy

            ply_strain_xy.append(eps_xy)
            ply_stress_xy.append(sig_xy)
            ply_stress_12.append(sig_12)
            ply_strain_12.append(eps_12)

        return {
            'eps0':          eps0,
            'kappa':         kappa,
            'ply_strain_xy': ply_strain_xy,
            'ply_stress_xy': ply_stress_xy,
            'ply_stress_12': ply_stress_12,
            'ply_strain_12': ply_strain_12,
        }

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def summary(self) -> str:
        lines = [
            f"Laminate  –  {len(self.plies)} plies  |  h = {self._h*1e3:.3f} mm",
            f"  Stacking: [{'/'.join(str(int(p.angle_deg)) for p in self.plies)}]",
            "",
            "  A matrix [MN/m]:",
        ]
        A_MPa = self.A / 1e6
        for row in A_MPa:
            lines.append("    " + "  ".join(f"{v:10.3f}" for v in row))
        lines += ["", "  D matrix [N·m]:"]
        for row in self.D:
            lines.append("    " + "  ".join(f"{v:10.3f}" for v in row))
        return "\n".join(lines)

    def __repr__(self) -> str:
        stack = "/".join(str(int(p.angle_deg)) for p in self.plies)
        return f"Laminate([{stack}], h={self._h*1e3:.3f} mm)"
