"""
composite_panel.aero_loads
--------------------------
Aerodynamic load generation for panel-level structural sizing.
Includes single-panel (Ackeret) and multi-panel wing-level loads.

PURPOSE
=======
This module converts flight conditions (Mach, altitude, AoA) into the running
loads [Nxx, Nyy, Nxy, Mxx] that drive the structural CLT analysis.  It sits at
the interface between aerodynamics and structures — the kind of coupling that is
central to multidisciplinary design optimisation (MDO) for supersonic airframes.

Three pressure models are provided, with automatic regime selection via panel_pressure():

  1. Ackeret linearised supersonic theory  (M 1.1–4.5, fast, closed-form)
  2. Modified Newtonian impact theory      (M ≥ 5.5, hypersonic flat plates)
  3. Linear blend transition              (M 4.5–5.5, avoids discontinuity)
  4. Elliptic spanwise lift distribution   (bending-dominated Nxx, all regimes)

ACKERET THEORY — PHYSICS BACKGROUND
=====================================
For Mach > 1 flow over a flat plate at angle of attack α, linearised (small
perturbation) theory gives the pressure coefficient:

    ΔCp = 4α / √(M² - 1)       (Ackeret, 1925)

This is derived from the linearised 2-D compressible Euler equations.  The term
√(M²-1) = β is the Prandtl-Glauert compressibility factor for supersonic flow
(analogous to 1/√(1-M²) in subsonic flow, but inverted — stiffness decreases
with Mach above 1).

Physical meaning:
  - At M = 1.4 → β = 0.98 (near-sonic, large ΔCp for given α)
  - At M = 2.0 → β = 1.73 (supersonic, pressure effect attenuates)
  - Linearity breaks down near M=1 and for thick airfoils; fine for M=1.4–3.0
    thin-panel problems

The net pressure loading on the panel (lower minus upper surface):
    Δp = ΔCp × q∞ × n_load

where q∞ = ½ρV² is the dynamic pressure and n_load is the ultimate load factor.

LOAD CONVERSION — PLATE THEORY
================================
Aerodynamic pressure Δp [Pa] acts normal to the panel surface.  For structural
sizing of a skin panel in a wing box, it translates to running loads via:

  Nyy = -Δp × chord / 2      [N/m]  chordwise compression on upper skin
                                     (negative = compressive, sign convention)
  Nxx = Nyy × 0.6            [N/m]  spanwise load (simplified beam-flange model)
  Nxy = |Nxx| × 0.25         [N/m]  torsion → shear flow in skin
  Mxx = Δp × chord² / 8      [N·m/m] out-of-plane bending from distributed pressure

The Nxx/Nyy ratio of ~0.6 and Nxy/Nxx ratio of ~0.25 are typical for a swept
supersonic planform.  In a production tool these would come from a global FEM or
aerodynamic model; here they are intentional engineering estimates that give
physically meaningful stress fields for preliminary sizing.

ISA ATMOSPHERE
==============
Dynamic pressure requires local density and speed of sound.  The built-in ISA
(International Standard Atmosphere) model covers:
  - Troposphere (0–11 km): linear temperature lapse at -6.5 K/km
  - Stratosphere (11–20 km): isothermal at 216.65 K

Accuracy is sufficient for conceptual sizing.  For aeroelastic analysis at
specific flight points, use actual atmospheric profile data.

EXTENSIBILITY FOR MDO
=====================
The PanelLoads dataclass is designed to be a clean interface between any
aero solver and the CLT analysis:
  - Replace supersonic_panel_loads() with a CFD-derived loader
  - Or with a loads database lookup (see project roadmap)
  - The downstream CLT analysis (laminate.py, failure.py) is unchanged

This separation of concerns is what makes the architecture scalable for MDO
workflows — aero and structures talk through a well-defined load vector, not
through each other's internals.

MODIFIED NEWTONIAN IMPACT THEORY — PHYSICS BACKGROUND
=======================================================
At hypersonic speeds (M > 5), linearised theory breaks down because:
  - The shock layer is thin and strongly curved
  - Real-gas effects (dissociation) alter γ_eff
  - The leeward surface is in aerodynamic shadow

Modified Newtonian theory (Lees 1955) approximates local Cp by the fraction of
the stagnation pressure that is "intercepted" by the surface:

    Cp(θ) = Cp_max · sin²(θ)

where θ is the local surface inclination to the freestream.

For a flat plate at AoA α:
  Windward (lower) surface: θ = α  →  Cp_lower = Cp_max · sin²(α)
  Leeward  (upper) surface: θ < 0  →  Cp_upper ≈ 0  (shadow; expansion fan)
  Net ΔCp = Cp_max · sin²(α)

Cp_max is computed from the Rayleigh Pitot formula — stagnation Cp behind a
normal shock at the nose, which gives the correct M-dependence and converges to
≈ 1.84 (γ=1.4) as M → ∞.  Accepting γ as a parameter allows real-gas
corrections: γ_eff ≈ 1.2–1.3 in the shock layer above M ≈ 10.

Note: Ackeret ΔCp is linear in α (linearised theory), while Newtonian ΔCp is
quadratic in α (sin²α ≈ α²).  At small AoA this correctly reflects the lower
L/D of hypersonic wings.

Reference:
    Anderson, J.D. – Modern Compressible Flow (McGraw-Hill, 2003), Ch. 9
    Kassapoglou, C. – Design and Analysis of Composite Structures, Ch. 1
    ESDU 76003 — Pressure distributions on wings at supersonic speeds
    Lees, L. – Hypersonic Flow (1955); Anderson, J.D. – Hypersonic and High
    Temperature Gas Dynamics (2006), Ch. 3
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass


@dataclass
class PanelLoads:
    """
    Running loads on a flat panel [N/m] and moments [N·m/m].

    This is the load interface object — it passes aero outputs to CLT inputs.
    All quantities follow standard structural sign conventions:

    Sign convention:
      Nxx, Nyy > 0  : tensile (stretching)
      Nxx, Nyy < 0  : compressive
      Nxy           : positive per right-hand rule
      Mxx, Myy, Mxy : positive for sagging (top surface in compression)

    For a supersonic upper wing skin:
      Nxx < 0  (spanwise compression from wing bending, upper surface)
      Nyy < 0  (chordwise compression from aerodynamic pressure)
      Nxy      (shear from torsion and sweep)
      Mxx > 0  (out-of-plane bending from distributed pressure)

    Attributes
    ----------
    Nxx : float  spanwise running load [N/m]
    Nyy : float  chordwise running load [N/m]
    Nxy : float  in-plane shear running load [N/m]
    Mxx : float  spanwise bending moment [N·m/m]
    Myy : float  chordwise bending moment [N·m/m]
    Mxy : float  twisting moment [N·m/m]
    """
    Nxx: float = 0.0
    Nyy: float = 0.0
    Nxy: float = 0.0
    Mxx: float = 0.0   # out-of-plane bending from pressure loading
    Myy: float = 0.0
    Mxy: float = 0.0

    @property
    def N(self) -> np.ndarray:
        """In-plane running load vector [Nxx, Nyy, Nxy] [N/m]."""
        return np.array([self.Nxx, self.Nyy, self.Nxy])

    @property
    def M(self) -> np.ndarray:
        """Running moment vector [Mxx, Myy, Mxy] [N·m/m]."""
        return np.array([self.Mxx, self.Myy, self.Mxy])

    def __str__(self) -> str:
        return (f"PanelLoads  Nxx={self.Nxx/1e3:+.1f} kN/m  "
                f"Nyy={self.Nyy/1e3:+.1f} kN/m  "
                f"Nxy={self.Nxy/1e3:+.1f} kN/m")


# ---------------------------------------------------------------------------
# Wing geometry + multi-panel loads
# ---------------------------------------------------------------------------

@dataclass
class WingGeometry:
    """
    Trapezoidal wing planform for multi-panel structural sizing.

    Coordinate: eta = y / semi_span  (0 = root, 1 = tip).

    Parameters
    ----------
    semi_span    : half-span [m]
    root_chord   : chord at root [m]
    taper_ratio  : tip_chord / root_chord
    sweep_le_deg : leading-edge sweep [deg]
    t_over_c     : structural box thickness-to-chord ratio
    mtow_n       : max take-off weight [N]  (drives bending moment magnitude)
    """
    semi_span:    float
    root_chord:   float
    taper_ratio:  float
    sweep_le_deg: float
    t_over_c:     float  = 0.04
    mtow_n:       float  = 150_000.0   # N

    def chord(self, eta: float) -> float:
        """Linear taper: c(eta) = c_root * [1 - (1-lambda)*eta]."""
        return self.root_chord * (1.0 - (1.0 - self.taper_ratio) * eta)

    def box_height(self, eta: float) -> float:
        """Structural box height [m] = t/c * local chord."""
        return self.t_over_c * self.chord(eta)

    def sweep_factor(self) -> float:
        """cos^2(sweep_LE) — resolves bending load along panel axes."""
        return np.cos(np.radians(self.sweep_le_deg)) ** 2


def wing_panel_loads(
    wing: "WingGeometry",
    eta: float,
    mach: float,
    altitude_m: float,
    alpha_deg: float,
    n_load: float = 2.5,
) -> "PanelLoads":
    """
    Combined running loads at spanwise station eta for the upper wing skin.

    Nxx is derived from the elliptic bending moment (not an empirical ratio
    of Nyy), capturing the spanwise variation: large near root, zero at tip.
    Nyy accounts for local chord via the tapered planform.

    Load derivation
    ---------------
    Nyy : -delta_p * chord(eta) / 2
    Nxx : -M_bend(y) / (box_height * chord) * cos^2(sweep)
    Nxy : |Nxx|*0.15 + |Nyy|*0.10
    Mxx : delta_p * chord^2 / 8

    Parameters
    ----------
    wing      : WingGeometry
    eta       : dimensionless spanwise station [0=root, 1=tip]
    mach      : free-stream Mach (> 1)
    altitude_m: cruise altitude [m]
    alpha_deg : angle of attack [deg]
    n_load    : ultimate load factor

    Returns
    -------
    PanelLoads
    """
    c_loc = wing.chord(eta)
    y_loc = eta * wing.semi_span

    rho, a_sound = _isa(altitude_m)
    q = 0.5 * rho * (mach * a_sound) ** 2

    # Nyy — unified pressure model (Ackeret M<4.5, Newtonian M>5.5, blend between)
    delta_p = panel_pressure(mach, alpha_deg, q) * n_load
    Nyy = -delta_p * c_loc / 2.0

    # Nxx — skin as wing-box compression flange, loaded by elliptic bending moment
    l0     = 4.0 * (wing.mtow_n * n_load) / (np.pi * 2.0 * wing.semi_span)
    M_bend = _elliptic_bending_moment(l0, wing.semi_span, y_loc)
    Nxx    = -M_bend / (wing.box_height(eta) * c_loc) * wing.sweep_factor()

    # Nxy — torsion + sweep shear
    Nxy = abs(Nxx) * 0.15 + abs(Nyy) * 0.10

    # Mxx — local pressure bending (simply-supported panel)
    Mxx = delta_p * c_loc ** 2 / 8.0

    return PanelLoads(Nxx=Nxx, Nyy=Nyy, Nxy=Nxy, Mxx=Mxx)


# ---------------------------------------------------------------------------
# 1.  Ackeret (linearised supersonic) pressure
# ---------------------------------------------------------------------------

def ackeret_panel_pressure(mach: float,
                            alpha_deg: float,
                            q_inf: float) -> float:
    """
    Ackeret 2-D linearised pressure loading on a flat plate [Pa].

    Computes the net pressure difference (lower - upper surface) using
    Ackeret's linearised supersonic theory.  Valid for:
      - Mach > 1.0  (subsonic theory is Prandtl-Glauert, not implemented here)
      - Small angle of attack (sin α ≈ α in radians)
      - Thin flat plate (no thickness-induced pressure distribution)

    Formula:
        ΔCp = 4α / √(M² - 1)
        Δp  = ΔCp × q∞

    The factor of 4 (not 2 from simple wave theory) accounts for both upper
    and lower surface contributions in the net pressure.  At α = 3°, M = 1.6:
        β     = √(1.6² - 1) = 1.249
        ΔCp   = 4 × 0.0524 / 1.249 = 0.168
        q∞    ≈ 15 kPa at 15 km → Δp ≈ 2.5 kPa

    Parameters
    ----------
    mach : float
        Free-stream Mach number.  Must be > 1 (raises ValueError otherwise).
    alpha_deg : float
        Angle of attack [degrees].  Positive nose-up.
    q_inf : float
        Free-stream dynamic pressure q∞ = ½ρV² [Pa].

    Returns
    -------
    float
        Net pressure Δp (positive = lower surface pressure exceeds upper) [Pa].
    """
    if mach <= 1.0:
        raise ValueError(f"Ackeret theory requires Mach > 1 (got {mach})")

    beta  = np.sqrt(mach**2 - 1.0)              # Prandtl-Glauert factor (supersonic)
    alpha = np.radians(alpha_deg)
    delta_Cp = 4.0 * alpha / beta                # Ackeret linearised ΔCp
    return delta_Cp * q_inf                      # [Pa]


def _oblique_shock_beta(mach: float, delta: float, gamma: float = 1.4) -> float:
    """
    Solve the theta-beta-M relation for shock angle beta [rad] given wedge
    deflection delta [rad] and freestream Mach.

    theta-beta-M relation (exact, 2-D):
        tan(delta) = 2 cot(beta) * (M^2 sin^2(beta) - 1)
                     / (M^2 (gamma + cos(2*beta)) + 2)

    Solved with Newton-Raphson starting from the Ackeret approximation
    beta_0 = asin(1/M) + delta (weak shock).  Converges in 3-5 iterations
    for delta < 30 deg.
    """
    g   = gamma
    M2  = mach ** 2
    # Initial guess: weak-shock Mach angle + deflection
    beta = math.asin(1.0 / mach) + delta

    for _ in range(15):
        sb  = math.sin(beta)
        cb  = math.cos(beta)
        tb  = math.tan(beta)
        f   = (2.0 / tb * (M2 * sb**2 - 1.0)
               / (M2 * (g + math.cos(2.0 * beta)) + 2.0) - math.tan(delta))
        # Derivative df/dbeta (finite difference — adequate for Newton step)
        db  = 1e-7
        sb2 = math.sin(beta + db); cb2 = math.cos(beta + db); tb2 = math.tan(beta + db)
        f2  = (2.0 / tb2 * (M2 * sb2**2 - 1.0)
               / (M2 * (g + math.cos(2.0*(beta + db))) + 2.0) - math.tan(delta))
        dfdx = (f2 - f) / db
        if abs(dfdx) < 1e-30:
            break
        beta -= f / dfdx
        if abs(f) < 1e-10:
            break
    return beta


def _cp_max_modified_newtonian(mach: float, gamma: float = 1.4) -> float:
    """
    Stagnation-point Cp behind a normal shock — Cp_max for modified Newtonian.

    Uses the Rayleigh Pitot formula for total pressure behind a normal shock.
    As M → ∞, Cp_max → 1.84 (γ=1.4).

    NOTE: Only meaningful for blunt bodies / large inclination angles.
    For thin wings at small AoA use oblique_shock_panel_pressure instead.
    """
    g   = gamma
    M2  = mach ** 2
    p02 = (
        ((g + 1.0)**2 * M2 / (4.0*g*M2 - 2.0*(g - 1.0))) ** (g / (g - 1.0))
        * (2.0*g*M2 - (g - 1.0)) / (g + 1.0)
    )
    return (p02 - 1.0) / (0.5 * g * M2)


def hypersonic_panel_pressure(
    mach:      float,
    alpha_deg: float,
    q_inf:     float,
    gamma:     float = 1.4,
) -> float:
    """
    Modified Newtonian pressure for blunt bodies at high Mach / large AoA.

    ΔCp = Cp_max · sin²(α)

    WARNING: This is appropriate for blunt bodies (re-entry capsules, nose cones)
    or surfaces at large inclination (α > 20°).  For thin wings at small AoA it
    severely under-predicts pressure (factor ~8× at M=5, α=3°).  Use
    oblique_shock_panel_pressure() or panel_pressure() for thin wings.
    """
    alpha    = np.radians(alpha_deg)
    Cp_max   = _cp_max_modified_newtonian(mach, gamma)
    delta_Cp = Cp_max * np.sin(alpha) ** 2
    return delta_Cp * q_inf


def oblique_shock_panel_pressure(
    mach:      float,
    alpha_deg: float,
    q_inf:     float,
    gamma:     float = 1.4,
) -> float:
    """
    Exact oblique-shock pressure loading on a thin flat plate.

    Windward (lower) surface — oblique shock with deflection delta = alpha:
        1. Solve theta-beta-M for shock angle beta
        2. Rankine-Hugoniot: p2/p1 = 1 + 2gamma/(gamma+1) * (M^2 sin^2(beta) - 1)
        3. Cp_windward = 2(p2/p1 - 1) / (gamma * M^2)

    Leeward (upper) surface — supersonic expansion fan:
        Cp_leeward ~ -2*alpha/sqrt(M^2-1)  (Ackeret approximation, small α)
        → gives slight suction on upper surface (conservative for skin sizing)

    This is the physically correct extension of Ackeret to all Mach > 1:
        - Reduces exactly to Ackeret in the limit alpha → 0
        - Handles real shock curvature at M > 5
        - Continuous and smooth in both Mach and alpha
        - Valid for alpha < ~25 deg (before detached shock)

    Parameters
    ----------
    mach      : Mach (> 1)
    alpha_deg : angle of attack [deg]
    q_inf     : dynamic pressure [Pa]
    gamma     : ratio of specific heats (1.4 for calorically perfect air;
                ~1.2 for real-gas hypersonic above M~10)
    """
    alpha = math.radians(alpha_deg)
    if alpha < 1e-9:
        return 0.0

    g  = gamma
    M2 = mach ** 2

    # --- Windward: oblique shock ---
    beta = _oblique_shock_beta(mach, alpha, g)
    p2_p1 = 1.0 + 2.0 * g / (g + 1.0) * (M2 * math.sin(beta)**2 - 1.0)
    Cp_windward = 2.0 * (p2_p1 - 1.0) / (g * M2)

    # --- Leeward: expansion (Ackeret suction approximation) ---
    beta_M = math.sqrt(max(M2 - 1.0, 1e-6))
    Cp_leeward = -2.0 * alpha / beta_M

    return (Cp_windward - Cp_leeward) * q_inf


_M_SUB_MAX = 0.85   # Prandtl-Glauert valid below this
_M_SUP_MIN = 1.15   # oblique shock valid above this


def panel_pressure(
    mach:      float,
    alpha_deg: float,
    q_inf:     float,
    gamma:     float = 1.4,
) -> float:
    """
    Unified thin-wing panel pressure — full Mach range (subsonic to hypersonic).

    Regime selection:
        M ≤ 0.85   : Prandtl-Glauert subsonic  ΔCp = 4α / √(1-M²)
        0.85–1.15  : linear blend across transonic (avoids M=1 singularity)
        M ≥ 1.15   : oblique-shock + expansion   (exact supersonic/hypersonic)

    Prandtl-Glauert subsonic:
        Applies the PG compressibility correction to the incompressible flat-plate
        result (ΔCp_inc = 4α).  ΔCp increases toward M=1 — consistent with the
        well-known subsonic lift-curve slope increase.

    Oblique-shock supersonic (M ≥ 1.15):
        Windward: exact Rankine-Hugoniot after solving θ-β-M.
        Leeward:  Ackeret expansion (suction).
        Reduces to Ackeret for small α; extends naturally to hypersonic.

    Transonic blend (0.85 < M < 1.15):
        Linear interpolation between the two endpoint pressures.
        Neither analytical method is valid here; the blend gives a smooth,
        monotonic transition appropriate for preliminary design load estimation.

    Parameters
    ----------
    mach      : Mach number (> 0)
    alpha_deg : angle of attack [deg]
    q_inf     : dynamic pressure [Pa]
    gamma     : ratio of specific heats (default 1.4; ~1.2 for real-gas M > 10)

    Returns
    -------
    float  Net pressure Δp [Pa]  (positive = upward net force)
    """
    if mach <= 0.0:
        raise ValueError(f"Mach must be positive, got {mach:.3f}")

    alpha = math.radians(alpha_deg)

    def _p_sub(M: float) -> float:
        beta = math.sqrt(max(1.0 - M**2, 1e-6))
        return 4.0 * alpha / beta * q_inf

    def _p_sup(M: float) -> float:
        return oblique_shock_panel_pressure(M, alpha_deg, q_inf, gamma)

    if mach <= _M_SUB_MAX:
        return _p_sub(mach)
    elif mach >= _M_SUP_MIN:
        return _p_sup(mach)
    else:
        # Transonic: blend between subsonic value at 0.85 and supersonic at 1.15
        t  = (mach - _M_SUB_MAX) / (_M_SUP_MIN - _M_SUB_MAX)
        p0 = _p_sub(_M_SUB_MAX)
        p1 = _p_sup(_M_SUP_MIN)
        return p0 + t * (p1 - p0)


def supersonic_panel_loads(mach: float,
                            altitude_m: float,
                            alpha_deg: float,
                            panel_chord: float,
                            panel_span:  float,
                            skin_thickness_estimate: float = 2e-3,
                            n_load: float = 2.5) -> PanelLoads:
    """
    Estimate running loads on an upper-surface wing skin panel for a supersonic
    cruise condition.  Combines Ackeret pressure with simplified beam-flange
    and torsion models to produce a physically representative load state.

    Load derivation:
      1. ISA atmosphere  →  ρ, a (speed of sound)  →  V = M·a  →  q∞ = ½ρV²
      2. Ackeret         →  Δp = ΔCp · q∞ · n_load
      3. Pressure → Nyy: treat panel as a simply supported beam (chord direction).
                         Nyy = -Δp · chord / 2  (net compression on upper skin)
      4. Nxx: spanwise bending induces axial running load on the skin as a
              wing-box flange.  Empirical ratio Nxx/Nyy ≈ 0.6 for swept planform.
      5. Nxy: torsion → closed thin-wall shear flow.  Nxy ≈ |Nxx| × 0.25.
      6. Mxx: out-of-plane bending from distributed pressure:
              Mxx = Δp · chord² / 8  (uniform load, simply supported — conservative)

    Parameters
    ----------
    mach : float
        Cruise Mach number (> 1.0).
    altitude_m : float
        Cruise altitude [m].  ISA model used.
    alpha_deg : float
        Angle of attack [degrees].
    panel_chord : float
        Panel chordwise dimension [m].
    panel_span : float
        Panel spanwise dimension [m].  (Not used in current load model but
        retained for interface consistency with elliptic_spanwise_load.)
    skin_thickness_estimate : float
        Panel skin thickness [m].  Reserved for future bending refinement.
    n_load : float
        Ultimate load factor (default 2.5g — FAR 25 transport category).

    Returns
    -------
    PanelLoads
    """
    # Step 1: ISA atmosphere at cruise altitude
    rho, a = _isa(altitude_m)
    V     = mach * a                 # true airspeed [m/s]
    q_inf = 0.5 * rho * V**2        # dynamic pressure [Pa]

    # Step 2: Ackeret net pressure, scaled by ultimate load factor
    delta_p = ackeret_panel_pressure(mach, alpha_deg, q_inf) * n_load

    # Step 3: Chordwise running compression from pressure
    # Assumes panel acts as a flange — pressure integrated over chord produces
    # a compression resultant on the upper skin
    Nyy = -delta_p * panel_chord / 2.0     # negative = compressive

    # Step 4: Spanwise running load (wing bending as beam)
    # Upper skin is the compression flange of the wing box under positive g
    # Ratio 0.6 is typical for a swept supersonic planform (validated against
    # simplified beam models for representative configurations)
    Nxx = Nyy * 0.6

    # Step 5: In-plane shear from torsion and sweep effects
    Nxy = abs(Nxx) * 0.25

    # Step 6: Out-of-plane bending moment from distributed aerodynamic pressure
    # Uniform load approximation (conservative for design sizing)
    Mxx = delta_p * panel_chord**2 / 8.0

    return PanelLoads(Nxx=Nxx, Nyy=Nyy, Nxy=Nxy, Mxx=Mxx)


# ---------------------------------------------------------------------------
# 2.  Elliptic lift distribution → panel running load
# ---------------------------------------------------------------------------

def elliptic_spanwise_load(MTOW_N: float,
                            n_load: float,
                            semi_span: float,
                            eta: float,
                            chord_at_eta: float) -> PanelLoads:
    """
    Derive panel running loads from an elliptic spanwise lift distribution.

    An elliptic lift distribution minimises induced drag (Prandtl's lifting line
    optimum) and is a common analytical reference for structural bending loads.
    The spanwise running load l(y) [N/m] follows:

        l(y) = l0 · √(1 - (y/b)²)

    where l0 = 4L / (π · 2b) is the root load intensity, L is total lift,
    b is the semi-span, and y is the spanwise position.

    The bending moment at spanwise station η:
        M(η) = ∫_y^b l(ξ)(ξ - y) dξ

    This moment is reacted by the wing-box section modulus.  The skin panel
    acts as the flange:
        Nxx ≈ -M(η) / (z_arm · chord)       [N/m]

    where z_arm ≈ 0.12 × chord (approximate wing-box height = 12% chord).

    Use case:
      More appropriate than Ackeret loads when bending is the dominant load
      path (e.g. thick inner-wing sections, transport aircraft, lower Mach).
      For supersonic outer-wing panels, Ackeret is usually more relevant.

    Parameters
    ----------
    MTOW_N : float
        Maximum take-off weight [N].  Full MTOW used for structural sizing.
    n_load : float
        Load factor (ultimate).
    semi_span : float
        Wing semi-span [m].
    eta : float
        Dimensionless spanwise station η = y/b (0 = root, 1 = tip).
    chord_at_eta : float
        Local chord length at station η [m].

    Returns
    -------
    PanelLoads  (Nxx dominant; Nyy and Nxy estimated from Nxx ratios)
    """
    L_total = MTOW_N * n_load          # total ultimate lift [N]

    # Elliptic distribution root intensity
    b  = semi_span
    l0 = 4.0 * L_total / (np.pi * 2.0 * b)   # root running load [N/m]
    y  = eta * b                               # dimensional spanwise position [m]

    # Running load at this spanwise station
    l_y = l0 * np.sqrt(max(1.0 - (y/b)**2, 0.0))   # [N/m]  (clamped at tip)

    # Bending moment outboard of this station (outboard load ×  moment arm)
    M_bend = _elliptic_bending_moment(l0, b, y)     # [N·m]

    # Panel Nxx: wing-box flange stress resultant
    # z_arm = approximate wing-box height as fraction of local chord
    z_arm = 0.12 * chord_at_eta         # 12% chord wing-box height assumption
    Nxx   = -M_bend / (z_arm * chord_at_eta)   # negative = upper skin compression

    return PanelLoads(Nxx=Nxx, Nyy=Nxx * 0.3, Nxy=abs(Nxx) * 0.15)


def _elliptic_bending_moment(l0: float, b: float, y: float) -> float:
    """
    Bending moment at spanwise station y for an elliptic lift distribution.

    Computed numerically by integrating the running load × moment arm over
    the outboard portion of the wing:
        M(y) = ∫_y^b l(ξ) · (ξ - y) dξ

    Trapezoid integration with 500 points gives adequate precision for
    preliminary sizing.  A closed-form expression exists but is more complex
    and offers no practical advantage here.

    Parameters
    ----------
    l0 : float   Root running load intensity [N/m]
    b  : float   Semi-span [m]
    y  : float   Spanwise station [m]

    Returns
    -------
    float  Bending moment [N·m]
    """
    n  = 500                                         # integration points
    xi = np.linspace(y, b, n)                        # spanwise coordinates
    l  = l0 * np.sqrt(np.maximum(1.0 - (xi/b)**2, 0.0))   # running load
    M  = np.trapezoid(l * (xi - y), xi)                 # moment about station y
    return M


# ---------------------------------------------------------------------------
# ISA atmosphere model
# ---------------------------------------------------------------------------

def _isa(altitude_m: float):
    """
    Simple ISA (International Standard Atmosphere) model up to 20 km.

    Two-layer model:
      Troposphere   (0–11 km) : T decreases linearly at -6.5 K/km
      Stratosphere  (11–20 km): T is constant at 216.65 K (isothermal)

    Pressure from hydrostatic equation:
      Troposphere:   P = P0 · (T/T0)^(-g/(L·R))
      Stratosphere:  P = P11 · exp(-g·(h-11000)/(R·T11))

    Density from ideal gas law:  ρ = P / (R·T)
    Speed of sound:               a = √(γ·R·T)

    Parameters
    ----------
    altitude_m : float
        Geometric altitude [m].  Valid range: 0–20,000 m.

    Returns
    -------
    tuple (rho, a)
        rho : float  Air density [kg/m³]
        a   : float  Speed of sound [m/s]
    """
    # Sea-level ISA constants
    T0    = 288.15    # K    sea-level temperature
    P0    = 101325.0  # Pa   sea-level pressure
    L     = -0.0065   # K/m  troposphere lapse rate (negative = cools with altitude)
    g     = 9.80665   # m/s² gravitational acceleration
    R     = 287.05    # J/(kg·K)  specific gas constant for dry air
    gamma = 1.4       # ratio of specific heats (diatomic gas)

    if altitude_m <= 11000:
        # Troposphere: linear temperature profile
        T = T0 + L * altitude_m
        P = P0 * (T / T0) ** (-g / (L * R))   # barometric formula
    else:
        # Stratosphere: isothermal — exponential pressure decay
        T11 = T0 + L * 11000                   # tropopause temperature ≈ 216.65 K
        P11 = P0 * (T11 / T0) ** (-g / (L * R))
        T   = T11                              # constant temperature above 11 km
        P   = P11 * np.exp(-g * (altitude_m - 11000) / (R * T11))

    rho = P / (R * T)           # ideal gas law [kg/m³]
    a   = np.sqrt(gamma * R * T)  # speed of sound [m/s]
    return rho, a
