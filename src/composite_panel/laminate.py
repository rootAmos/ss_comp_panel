"""
composite_panel.laminate
------------------------
Classical Laminate Theory (CLT) implementation.

THEORY OVERVIEW
===============
CLT extends the Euler-Bernoulli beam and Kirchhoff plate assumptions to
layered anisotropic media.  The key kinematic assumption is that plane sections
remain plane after deformation — i.e., through-thickness strain varies linearly:

    ε(z) = ε0 + z · κ

where:
  ε0   = midplane strains  [εx0, εy0, γxy0]    — driven by in-plane loads N
  κ    = curvatures        [κx,  κy,  κxy ]     — driven by bending moments M
  z    = distance from mid-plane                — ranges over [-h/2, +h/2]

THE ABD MATRIX
==============
Integrating the ply stresses through the laminate thickness gives the
constitutive relation at the laminate level:

    [ N ]   [ A  B ] [ ε0 ]
    [ M ] = [ B  D ] [ κ  ]

where N = [Nxx, Nyy, Nxy] are running loads [N/m] and M = [Mxx, Myy, Mxy]
are running moments [N·m/m].  The A, B, D submatrices are:

    A_ij = Σ_k  Q_bar_ij^(k) · (z_{k+1} - z_k)               [N/m]
    B_ij = Σ_k  Q_bar_ij^(k) · (z_{k+1}² - z_k²) / 2         [N]
    D_ij = Σ_k  Q_bar_ij^(k) · (z_{k+1}³ - z_k³) / 3         [N·m]

Physical meaning:
  A : extensional stiffness    — relates in-plane loads to mid-plane strains
  B : coupling stiffness       — non-zero only for UNSYMMETRIC laminates;
                                  causes bending when loaded in-plane (warping)
  D : bending stiffness        — relates moments to curvatures

DESIGN RULES FROM CLT
=====================
Symmetric laminates (stacking symmetric about mid-plane) have B = 0, which
eliminates bending-extension coupling.  This is almost always required for
flight structures to avoid thermally induced warpage during cure.

Balanced laminates (+θ and -θ plies in equal numbers) have A16 = A26 = 0,
which eliminates shear-extension coupling — important for straight-running
loads without unexpected shear distortion.

Quasi-isotropic laminates (e.g. [±45/0/90]s) have an isotropic A matrix
(Ex = Ey, A16 = A26 = 0), making them efficient for multi-directional load
fields typical of fuselage panels and wing skins near root fittings.

SOLUTION SEQUENCE
=================
Given loads [N, M]:
  1. Invert ABD → compliance abd
  2. Solve  [ε0, κ] = abd @ [N, M]
  3. At each ply mid-plane z_mid: ε_xy = ε0 + z_mid · κ
  4. σ_xy = Q_bar @ ε_xy                (stress in laminate frame)
  5. σ_12 = T @ σ_xy                    (rotate to ply principal axes)
  6. Apply failure criterion to σ_12    (see failure.py)

Reference:
    Kassapoglou, C. – Design and Analysis of Composite Structures
    (Wiley, 2013), Ch. 3–4

    Jones, R.M. – Mechanics of Composite Materials (Taylor & Francis, 1999), Ch. 4
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional

from .ply import Ply


class Laminate:
    """
    Composite laminate defined by an ordered stack of Ply objects.

    The laminate coordinate system has z = 0 at the mid-plane, with z
    increasing upward (toward the outer skin surface in a wing panel).
    Plies are ordered BOTTOM (z = -h/2) to TOP (z = +h/2), matching the
    physical layup sequence as seen during hand-layup or automated fibre
    placement (AFP).

    ABD computation is lazy — the 6×6 matrix is only assembled on first access,
    so constructing many Laminate objects for a trade study (e.g. stacking angle
    sweep) is cheap until a property is actually queried.

    Parameters
    ----------
    plies : list of Ply
        Ordered from BOTTOM (z = -h/2) to TOP (z = +h/2).
        For a symmetric layup [A/B/C]s, pass all 2n plies explicitly.

    Notes
    -----
    z = 0 is the laminate mid-plane.  The coordinate is needed for the B and D
    integrals — physically, D grows as z³, so outer plies contribute most to
    bending stiffness.  This is why carbon fibres are placed on the outer faces
    of sandwich panels and hybrid laminates.
    """

    def __init__(self, plies: List[Ply]):
        self.plies = plies
        self._build_z_coords()

    # ------------------------------------------------------------------
    # Geometry — ply interface z-coordinates
    # ------------------------------------------------------------------

    def _build_z_coords(self):
        """
        Compute the z-coordinate of each ply interface and precompute
        per-ply arrays used in response().

        The z-axis runs from -h/2 (bottom of laminate) to +h/2 (top).
        Interface z_k is the bottom face of ply k; z_{k+1} is the top face.

        These are used in the thickness-integration for A, B, D:
            Δz  = z_{k+1} - z_k      → contributes to A
            Δz² = z_{k+1}² - z_k²    → contributes to B
            Δz³ = z_{k+1}³ - z_k³    → contributes to D

        Stacked ply matrices (Qbar, T, T_strain) are assembled here once so
        response() can operate without any Python-level ply loop.
        """
        thicknesses = np.array([p.thickness for p in self.plies])
        total = float(thicknesses.sum())
        self._h = total           # total laminate thickness [m]

        # z[0] = -h/2 (bottom face of ply 0), accumulated with cumsum
        self._z = np.empty(len(self.plies) + 1)
        self._z[0] = -total / 2.0
        self._z[1:] = -total / 2.0 + np.cumsum(thicknesses)

        # Ply midplane z-coordinates — needed every response() call
        self._z_mids = (self._z[:-1] + self._z[1:]) / 2.0  # shape (n,)

        # Stacked (n, 3, 3) ply matrices — assembled once, reused every call
        self._Qbar_stack   = np.stack([p.Q_bar    for p in self.plies])
        self._T_stack      = np.stack([p.T        for p in self.plies])
        self._T_strain_stack = np.stack([p.T_strain for p in self.plies])

    @property
    def thickness(self) -> float:
        """Total cured laminate thickness [m]."""
        return self._h

    @property
    def z_interfaces(self) -> np.ndarray:
        """
        z-coordinates of all ply interfaces [m].

        Length = n_plies + 1.  z_interfaces[0] = -h/2 (laminate bottom),
        z_interfaces[-1] = +h/2 (laminate top).
        """
        return self._z

    # ------------------------------------------------------------------
    # ABD stiffness matrix
    # ------------------------------------------------------------------

    def _compute_ABD(self):
        """
        Assemble the 6×6 ABD stiffness matrix by numerical integration
        through the ply stack.

        Each ply contributes:
          A += Q_bar * Δz        (zeroth moment of stiffness — in-plane)
          B += Q_bar * Δz²/2     (first moment  — bending-extension coupling)
          D += Q_bar * Δz³/3     (second moment — bending)

        The ABD matrix is then assembled as a block matrix and inverted to
        give the compliance (abd), which maps loads → deformations.

        Why invert once and store?
            Inverting a 6×6 matrix is O(n³) but only needs to happen once per
            laminate.  All subsequent response() calls just do a matrix-vector
            multiply: O(n²).  This matters enormously in trade studies that
            call response() thousands of times.
        """
        z0 = self._z[:-1]   # bottom faces of each ply, shape (n,)
        z1 = self._z[1:]    # top faces of each ply,    shape (n,)

        dz  = z1 - z0                       # shape (n,)
        dz2 = (z1**2 - z0**2) / 2.0        # shape (n,)
        dz3 = (z1**3 - z0**3) / 3.0        # shape (n,)

        # A += Qbar[k] * dz[k]   for all k simultaneously via einsum
        A = np.einsum('kij,k->ij', self._Qbar_stack, dz)
        B = np.einsum('kij,k->ij', self._Qbar_stack, dz2)
        D = np.einsum('kij,k->ij', self._Qbar_stack, dz3)

        self._A = A
        self._B = B
        self._D = D

        # Assemble the full 6×6 stiffness matrix
        self._ABD = np.block([[A, B],
                               [B, D]])

        # Invert once — all response() calls reuse this
        self._abd = np.linalg.inv(self._ABD)

    # ------------------------------------------------------------------
    # Public stiffness matrix properties (lazy evaluation)
    # ------------------------------------------------------------------

    @property
    def A(self) -> np.ndarray:
        """
        3×3 extensional stiffness matrix [N/m].

        Maps midplane strains to in-plane running loads:  N = A @ ε0  (for B=0).

        Key entries:
          A11 : axial stiffness in x        — dominated by 0° plies
          A22 : axial stiffness in y        — dominated by 90° plies
          A66 : shear stiffness             — dominated by ±45° plies
          A16, A26 : shear-extension coupling (zero for balanced laminates)
        """
        if not hasattr(self, '_A'):
            self._compute_ABD()
        return self._A

    @property
    def B(self) -> np.ndarray:
        """
        3×3 bending-extension coupling stiffness matrix [N].

        Non-zero only for UNSYMMETRIC laminates.  For symmetric laminates
        B = 0 identically (odd-power z integrals vanish).

        Non-zero B causes:
          - Warping under temperature changes during cure (spring-back)
          - In-plane loads that produce bending (dangerous for panels)
        Symmetric laminates are strongly preferred for all flight structures.
        """
        if not hasattr(self, '_B'):
            self._compute_ABD()
        return self._B

    @property
    def D(self) -> np.ndarray:
        """
        3×3 bending stiffness matrix [N·m].

        Maps curvatures to bending moments:  M = D @ κ  (for B=0).

        Key entries:
          D11 : bending stiffness about y-axis  — 0° plies on outer faces maximise this
          D22 : bending stiffness about x-axis  — 90° plies on outer faces
          D66 : twisting stiffness              — ±45° plies on outer faces
          D16, D26 : bend-twist coupling (non-zero for unbalanced or off-axis laminates)

        D scales as z³, so placing stiff plies on the outer faces is exponentially
        more effective than the same plies near the mid-plane — this is the same
        principle as an I-beam flange.
        """
        if not hasattr(self, '_D'):
            self._compute_ABD()
        return self._D

    @property
    def ABD(self) -> np.ndarray:
        """
        6×6 full laminate stiffness matrix.

        Block structure:
            [ A  B ]
            [ B  D ]

        Used directly when bending-extension coupling (B ≠ 0) is present,
        or for export to finite element pre-processors.
        """
        if not hasattr(self, '_ABD'):
            self._compute_ABD()
        return self._ABD

    @property
    def abd(self) -> np.ndarray:
        """
        6×6 laminate compliance matrix (inverse of ABD).

        Maps applied loads to deformations:
            [ε0, κ] = abd @ [N, M]

        Partitioning for symmetric laminates (B = 0):
            a = A^-1     (extensional compliance [m/N])
            d = D^-1     (bending compliance [m/(N·m)])
        """
        if not hasattr(self, '_abd'):
            self._compute_ABD()
        return self._abd

    # ------------------------------------------------------------------
    # Effective engineering constants
    # ------------------------------------------------------------------

    @property
    def Ex(self) -> float:
        """
        Effective in-plane Young's modulus in x-direction [Pa].

        Derived from the extensional compliance:  Ex = 1 / (a11 · h)

        Valid for symmetric laminates where a = A^-1.  For unsymmetric
        laminates the coupling with bending must be accounted for.
        """
        return 1.0 / (self.abd[0, 0] * self._h)

    @property
    def Ey(self) -> float:
        """Effective in-plane Young's modulus in y-direction [Pa]."""
        return 1.0 / (self.abd[1, 1] * self._h)

    @property
    def Gxy(self) -> float:
        """Effective in-plane shear modulus [Pa]."""
        return 1.0 / (self.abd[2, 2] * self._h)

    def alpha_lam(self, ply_thermals) -> np.ndarray:
        """
        Effective laminate coefficient of thermal expansion [1/K].

        Returns [αx, αy, αxy] — the laminate-level CTEs in the global
        x-y frame, defined as the free thermal strains per unit ΔT for a
        mechanically unconstrained symmetric laminate (B = 0):

            [αx, αy, αxy] = A⁻¹ @ N_T(ΔT=1)

        where N_T is the thermal force resultant per unit temperature rise:

            N_T = Σ_k  Q̄_k @ ᾱ_k * t_k

        and ᾱ_k = [α₁c² + α₂s², α₁s² + α₂c², 2(α₁−α₂)cs] is the
        CTE vector of ply k transformed to laminate axes.

        Parameters
        ----------
        ply_thermals : list of PlyThermal
            One entry per ply, in the same order as self.plies.
            Use IM7_8552_thermal() to get the standard IM7/8552 values.

        Returns
        -------
        alpha : np.ndarray, shape (3,)
            [αx, αy, αxy] in 1/K.  For balanced symmetric laminates αxy ≈ 0.

        Notes
        -----
        For cross-ply laminates αx and αy lie between α₁ (fibre, ~0.3 µ/K)
        and α₂ (transverse, ~28.8 µ/K).  For quasi-isotropic laminates
        αx ≈ αy (isotropic thermal expansion).
        """
        import math
        from .thermal import alpha_bar

        NT_unit = np.zeros(3)  # thermal force resultant per unit ΔT
        for k, (ply, pt) in enumerate(zip(self.plies, ply_thermals)):
            alpha_xy = alpha_bar(pt, math.radians(ply.angle_deg))
            dz = float(self._z[k + 1] - self._z[k])
            NT_unit += ply.Q_bar @ alpha_xy * dz

        # a = A⁻¹ (extensional compliance, valid for symmetric B=0 laminates)
        a = np.linalg.inv(self.A)
        return a @ NT_unit

    # ------------------------------------------------------------------
    # Load-deformation response
    # ------------------------------------------------------------------

    def response(self,
                 N: Optional[np.ndarray] = None,
                 M: Optional[np.ndarray] = None) -> dict:
        """
        Compute the full CLT response for a given load state.

        This is the core analysis call.  Given running loads N and moments M,
        it returns midplane strains, curvatures, and ply-level stress/strain
        in both the laminate (x-y) and principal ply (1-2) coordinate frames.

        Solution sequence:
          1. Solve   [ε0, κ] = abd @ [N, M]
          2. For each ply at z = z_mid:
               ε_xy = ε0 + z_mid · κ         (kinematic — strain through thickness)
               σ_xy = Q_bar @ ε_xy            (constitutive — stress in global axes)
               σ_12 = T @ σ_xy                (rotate to ply fibre axes)
               ε_12 = T_strain @ ε_xy         (rotate strain to ply axes)

        The ply-axis stresses [σ1, σ2, τ12] are what failure criteria act on.

        Parameters
        ----------
        N : array-like, shape (3,) [N/m]
            Running in-plane loads  [Nxx, Nyy, Nxy].  Zero if not supplied.
        M : array-like, shape (3,) [N·m/m]
            Running bending moments [Mxx, Myy, Mxy].  Zero if not supplied.

        Returns
        -------
        dict with keys:
            'eps0'          : midplane strains   shape (3,)   [ε_x, ε_y, γ_xy]
            'kappa'         : curvatures         shape (3,)   [κ_x, κ_y, κ_xy]
            'ply_strain_xy' : list of (3,) – strain  at mid-ply z in laminate axes
            'ply_stress_xy' : list of (3,) – stress  at mid-ply z in laminate axes [Pa]
            'ply_stress_12' : list of (3,) – stress  in ply principal axes [Pa]
            'ply_strain_12' : list of (3,) – strain  in ply principal axes
        """
        N = np.zeros(3) if N is None else np.asarray(N, dtype=float)
        M = np.zeros(3) if M is None else np.asarray(M, dtype=float)

        # Solve the 6×6 system: [ε0, κ] = abd @ [N, M]
        load_vec = np.concatenate([N, M])
        deform   = self.abd @ load_vec
        eps0     = deform[:3]   # midplane strains  [εx0, εy0, γxy0]
        kappa    = deform[3:]   # curvatures        [κx,  κy,  κxy ]

        # ── Vectorised ply stress/strain recovery ────────────────────────────
        # eps_xy[k] = eps0 + z_mids[k] * kappa   →  shape (n, 3)
        eps_xy = eps0[None, :] + self._z_mids[:, None] * kappa[None, :]

        # sig_xy[k] = Qbar[k] @ eps_xy[k]         →  shape (n, 3)
        sig_xy = np.einsum('kij,kj->ki', self._Qbar_stack, eps_xy)

        # Rotate to ply (1-2) axes                →  shape (n, 3)
        sig_12 = np.einsum('kij,kj->ki', self._T_stack,       sig_xy)
        eps_12 = np.einsum('kij,kj->ki', self._T_strain_stack, eps_xy)

        return {
            'eps0':          eps0,
            'kappa':         kappa,
            'ply_strain_xy': eps_xy,    # (n, 3) — row k is ply k
            'ply_stress_xy': sig_xy,
            'ply_stress_12': sig_12,
            'ply_strain_12': eps_12,
        }

    # ------------------------------------------------------------------
    # Convenience / display
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """
        Return a formatted string with key laminate properties.

        Prints the A and D matrices in engineering units (MN/m and N·m),
        which are the most physically interpretable for structural sizing.
        B matrix is omitted — for symmetric laminates it is identically zero.
        """
        lines = [
            f"Laminate  –  {len(self.plies)} plies  |  h = {self._h*1e3:.3f} mm",
            f"  Stacking: [{'/'.join(str(int(p.angle_deg)) for p in self.plies)}]",
            "",
            "  A matrix [MN/m]:",
        ]
        A_MPa = self.A / 1e6   # convert N/m → MN/m for readability
        for row in A_MPa:
            lines.append("    " + "  ".join(f"{v:10.3f}" for v in row))
        lines += ["", "  D matrix [N·m]:"]
        for row in self.D:
            lines.append("    " + "  ".join(f"{v:10.3f}" for v in row))
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Laminate classification
    # ------------------------------------------------------------------

    @property
    def is_symmetric(self) -> bool:
        """
        True if the ply stack is symmetric about the mid-plane.

        A laminate is symmetric when the sequence of (angle, thickness) tuples
        is a palindrome.  Symmetric laminates have B = 0 identically.

        Angles are compared to within 0.1° and thicknesses to within 1 µm.
        """
        n = len(self.plies)
        for k in range(n // 2):
            p_bot = self.plies[k]
            p_top = self.plies[n - 1 - k]
            if (abs(p_bot.angle_deg - p_top.angle_deg) > 0.1 or
                    abs(p_bot.thickness - p_top.thickness) > 1e-6):
                return False
        return True

    @property
    def is_balanced(self) -> bool:
        """
        True if the laminate is balanced (A16 = A26 ≈ 0).

        A laminate is balanced when every off-axis ply at +θ has a
        corresponding −θ ply of equal thickness.  Equivalently,
        |A16| and |A26| are both below 0.1 % of sqrt(A11·A66).

        Non-zero A16/A26 causes shear-extension coupling (undesirable for
        straight-running loads but exploitable for aeroelastic tailoring).
        """
        A = self.A
        scale = float(np.sqrt(abs(A[0, 0] * A[2, 2]))) + 1e-30
        return (abs(A[0, 2]) / scale < 1e-3 and
                abs(A[1, 2]) / scale < 1e-3)

    @property
    def lamination_params(self) -> dict:
        """
        Compute the 12 lamination parameters for this laminate.

        Returns a dict with keys xi1A..xi4A, xi1B..xi4B, xi1D..xi4D.
        See lamination_parameters.lamination_parameters() for full docs.
        """
        from .lamination_parameters import lamination_parameters as _lp
        return _lp(self)

    def __repr__(self) -> str:
        stack = "/".join(str(int(p.angle_deg)) for p in self.plies)
        return f"Laminate([{stack}], h={self._h*1e3:.3f} mm)"
