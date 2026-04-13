"""
CLT implementation: ABD matrix assembly, laminate response, effective moduli.

Ref: Jones (1999) Ch. 4, Kassapoglou (2013) Ch. 3-4
"""

from __future__ import annotations

import aerosandbox.numpy as np
from typing import List, Optional

from composite_panel.ply import Ply


class Laminate:
    """
    Composite laminate from an ordered stack of Ply objects.

    Plies ordered bottom (z=-h/2) to top (z=+h/2).
    ABD computed lazily on first property access.
    """

    def __init__(self, plies: List[Ply]):
        self.plies = plies
        self._build_z_coords()

    def _build_z_coords(self):
        thicknesses = np.array([p.thickness for p in self.plies])
        total = float(thicknesses.sum())
        self._h = total
        self._thicknesses = thicknesses
        self._angles_deg = np.array([p.angle_deg for p in self.plies])

        self._z = np.empty(len(self.plies) + 1)
        self._z[0] = -total / 2.0
        self._z[1:] = -total / 2.0 + np.cumsum(thicknesses)

        self._z_mids = (self._z[:-1] + self._z[1:]) / 2.0

        self._Qbar_stack   = np.stack([p.Q_bar    for p in self.plies])
        self._T_stack      = np.stack([p.T        for p in self.plies])
        self._T_strain_stack = np.stack([p.T_strain for p in self.plies])

    @property
    def thickness(self) -> float:
        """Total laminate thickness [m]."""
        return self._h

    @property
    def z_interfaces(self) -> np.ndarray:
        """Ply interface z-coords [m], length n_plies+1."""
        return self._z

    # ------------------------------------------------------------------
    # ABD
    # ------------------------------------------------------------------

    def _compute_ABD(self):
        """Jones (1999), Eqs. 4.20-4.22."""
        z0 = self._z[:-1]
        z1 = self._z[1:]

        dz  = z1 - z0                    # A: Eq. 4.20
        dz2 = (z1**2 - z0**2) / 2.0      # B: Eq. 4.21
        dz3 = (z1**3 - z0**3) / 3.0      # D: Eq. 4.22

        A = np.einsum('kij,k->ij', self._Qbar_stack, dz)
        B = np.einsum('kij,k->ij', self._Qbar_stack, dz2)
        D = np.einsum('kij,k->ij', self._Qbar_stack, dz3)

        self._A = A
        self._B = B
        self._D = D
        self._ABD = np.block([[A, B], [B, D]])
        self._abd = np.linalg.inv(self._ABD)

    @property
    def A(self) -> np.ndarray:
        """3x3 extensional stiffness [N/m]."""
        if not hasattr(self, '_A'):
            self._compute_ABD()
        return self._A

    @property
    def B(self) -> np.ndarray:
        """3x3 coupling stiffness [N].  Zero for symmetric laminates."""
        if not hasattr(self, '_B'):
            self._compute_ABD()
        return self._B

    @property
    def D(self) -> np.ndarray:
        """3x3 bending stiffness [N*m]."""
        if not hasattr(self, '_D'):
            self._compute_ABD()
        return self._D

    @property
    def ABD(self) -> np.ndarray:
        """6x6 laminate stiffness."""
        if not hasattr(self, '_ABD'):
            self._compute_ABD()
        return self._ABD

    @property
    def abd(self) -> np.ndarray:
        """6x6 compliance (inverse of ABD)."""
        if not hasattr(self, '_abd'):
            self._compute_ABD()
        return self._abd

    # ------------------------------------------------------------------
    # Effective moduli
    # ------------------------------------------------------------------

    @property
    def Ex(self) -> float:
        """Effective in-plane Young's modulus in x [Pa].  Jones (1999), Eq. 4.37."""
        return 1.0 / (self.abd[0, 0] * self._h)

    @property
    def Ey(self) -> float:
        """Effective in-plane Young's modulus in y [Pa].  Jones (1999), Eq. 4.38."""
        return 1.0 / (self.abd[1, 1] * self._h)

    @property
    def Gxy(self) -> float:
        """Effective in-plane shear modulus [Pa].  Jones (1999), Eq. 4.39."""
        return 1.0 / (self.abd[2, 2] * self._h)

    def alpha_lam(self, ply_thermals) -> np.ndarray:
        """
        Laminate CTE [1/K] as [alphax, alphay, alphaxy].
        Valid for symmetric laminates (B=0).
        """
        alpha_1 = np.array([pt.alpha_1 for pt in ply_thermals])
        alpha_2 = np.array([pt.alpha_2 for pt in ply_thermals])
        angles_rad = np.radians(self._angles_deg)
        c = np.cos(angles_rad)
        s = np.sin(angles_rad)
        alpha_stack = np.stack([
            alpha_1 * c**2 + alpha_2 * s**2,
            alpha_1 * s**2 + alpha_2 * c**2,
            2.0 * (alpha_1 - alpha_2) * c * s,
        ], axis=1)
        dz = self._thicknesses

        # NT = Σ Q̄_k · α_k · Δz_k  →  einsum over ply stack
        NT_unit = np.einsum('kij,kj,k->i', self._Qbar_stack, alpha_stack, dz)

        return np.linalg.inv(self.A) @ NT_unit

    # ------------------------------------------------------------------
    # Response
    # ------------------------------------------------------------------

    def response(self,
                 N: Optional[np.ndarray] = None,
                 M: Optional[np.ndarray] = None) -> dict:
        """
        CLT response: midplane strains, curvatures, ply stresses/strains.

        Returns dict with keys: eps0, kappa, ply_strain_xy, ply_stress_xy,
        ply_stress_12, ply_strain_12.
        """
        N = np.zeros(3) if N is None else np.asarray(N, dtype=float)
        M = np.zeros(3) if M is None else np.asarray(M, dtype=float)

        load_vec = np.concatenate([N, M])
        deform   = self.abd @ load_vec
        eps0     = deform[:3]
        kappa    = deform[3:]

        eps_xy = eps0[None, :] + self._z_mids[:, None] * kappa[None, :]
        sig_xy = np.einsum('kij,kj->ki', self._Qbar_stack, eps_xy)
        sig_12 = np.einsum('kij,kj->ki', self._T_stack,       sig_xy)
        eps_12 = np.einsum('kij,kj->ki', self._T_strain_stack, eps_xy)

        return {
            'eps0':          eps0,
            'kappa':         kappa,
            'ply_strain_xy': eps_xy,
            'ply_stress_xy': sig_xy,
            'ply_stress_12': sig_12,
            'ply_strain_12': eps_12,
        }

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def summary(self) -> str:
        lines = [
            f"Laminate  -  {len(self.plies)} plies  |  h = {self._h*1e3:.3f} mm",
            f"  Stacking: [{'/'.join(str(int(p.angle_deg)) for p in self.plies)}]",
            "",
            "  A matrix [MN/m]:",
        ]
        A_MPa = self.A / 1e6
        for row in A_MPa:
            lines.append("    " + "  ".join(f"{v:10.3f}" for v in row))
        lines += ["", "  D matrix [N*m]:"]
        for row in self.D:
            lines.append("    " + "  ".join(f"{v:10.3f}" for v in row))
        return "\n".join(lines)

    @property
    def is_symmetric(self) -> bool:
        return bool(
            np.all(np.abs(self._angles_deg - self._angles_deg[::-1]) < 0.1)
            and np.all(np.abs(self._thicknesses - self._thicknesses[::-1]) < 1e-6)
        )

    @property
    def is_balanced(self) -> bool:
        A = self.A
        scale = float(np.sqrt(abs(A[0, 0] * A[2, 2]))) + 1e-30
        return (abs(A[0, 2]) / scale < 1e-3 and
                abs(A[1, 2]) / scale < 1e-3)

    def __repr__(self) -> str:
        stack = "/".join(str(int(p.angle_deg)) for p in self.plies)
        return f"Laminate([{stack}], h={self._h*1e3:.3f} mm)"


if __name__ == "__main__":
    import sys as _sys, os as _os
    _sys.stdout.reconfigure(encoding="utf-8")
    _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
    from composite_panel.ply import Ply, IM7_8552

    mat   = IM7_8552()
    t_ply = 0.125e-3

    angles = [0, 45, -45, 90, 90, -45, 45, 0]
    plies  = [Ply(mat, t_ply, a) for a in angles]
    lam    = Laminate(plies)

    print(lam.summary())
    print()
    print(f"Effective moduli:")
    print(f"  Ex  = {lam.Ex/1e9:.2f} GPa")
    print(f"  Ey  = {lam.Ey/1e9:.2f} GPa")
    print(f"  Gxy = {lam.Gxy/1e9:.2f} GPa")
    print()

    N = np.array([-280e3, -115e3, 42e3])
    M = np.array([   60.0,   0.0,  0.0])

    res = lam.response(N=N, M=M)
    eps0 = res['eps0']
    print(f"Applied N = [{N[0]/1e3:.0f}, {N[1]/1e3:.0f}, {N[2]/1e3:.0f}] kN/m  "
          f"M = [{M[0]:.0f}, {M[1]:.0f}, {M[2]:.0f}] N*m/m")
    print(f"Midplane strains (ueps):  epsx={eps0[0]*1e6:.1f},  epsy={eps0[1]*1e6:.1f},  "
          f"gammaxy={eps0[2]*1e6:.1f}")
