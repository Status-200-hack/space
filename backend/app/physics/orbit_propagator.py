"""
orbit_propagator.py
===================
Orbital propagation engine for the Autonomous Constellation Manager.

Coordinate system
-----------------
All positions and velocities are expressed in the **Earth-Centred Inertial
(ECI) J2000** frame.
  • Position : kilometres  [km]
  • Velocity : kilometres per second  [km/s]
  • Time     : seconds  [s]

Physics modelled
----------------
1. Point-mass two-body gravity  (μ/r²)
2. J2 zonal harmonic – Earth oblateness  (1st-order secular precession of
   RAAN and argument of perigee, plus radial acceleration corrections)
3. Atmospheric drag  (exponential scale-height density model, USSA76 fitted)

Numerical integration
---------------------
Classic **Runge-Kutta 4th-order** (RK4) fixed-step integrator.
The RK4 scheme achieves O(h⁴) local truncation error per step, which is
sufficient for short-arc propagation (minutes to days) with a step size of
O(10–60 seconds).

Public API
----------
  propagate_state(state, dt, ...)  → PropagationResult
      Advance a single StateVector by one time step dt.

  simulate_orbit(obj, time_seconds, ...) → OrbitSimulationResult
      Propagate a SimSatellite or SimDebris for a total duration, collecting
      state snapshots at every step.

Both functions are pure (no registry side-effects) and are safe to call
from background tasks or async endpoints.

Constants
---------
  μ     = 398 600.4418  km³/s²   (Earth gravitational parameter, EGM96)
  Rₑ    = 6 371.0       km       (mean Earth radius)
  J2    = 1.082 63 × 10⁻³       (second zonal harmonic)
  Rₑ_eq = 6 378.137     km       (WGS84 equatorial radius, used in J2 term)
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Union

import numpy as np

from app.models.sim_objects import SimSatellite, SimDebris, StateVector

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Physical constants  (all in km / s / kg)
# ═══════════════════════════════════════════════════════════════════════════

MU:      float = 398_600.4418   # km³ s⁻²   Earth gravitational parameter
RE:      float = 6_371.0        # km         mean Earth radius
RE_EQ:   float = 6_378.137      # km         WGS84 equatorial radius (J2 term)
J2:      float = 1.08263e-3     # –          second zonal harmonic
G0:      float = 9.80665e-3     # km s⁻²    standard gravity (for Isp → ve)

# ── Atmospheric density layers (USSA76 exponential fit) ─────────────────────
#    (base_alt_km, scale_height_km, rho_0 kg/m³)
_ATM: list = [
    (0,    8.44,  1.225_000e+00),
    (100,  5.88,  5.604_000e-07),
    (200,  7.29,  2.789_000e-10),
    (300,  8.34,  1.916_000e-12),
    (400,  7.51,  2.803_000e-13),
    (500,  8.66,  5.215_000e-14),
    (600, 11.04,  8.942_000e-15),
    (700, 16.56,  3.170_000e-15),
    (800, 27.27,  1.492_000e-15),
]


def _atm_density_km3(alt_km: float) -> float:
    """
    Exponential atmospheric density [kg/km³] for a given altitude.

    Converts from kg/m³ using 1 km³ = 1 e9 m³, so ρ[kg/km³] = ρ[kg/m³] × 1e9.
    Returns 0 above 1 000 km.
    """
    if alt_km >= 1_000.0 or alt_km < 0.0:
        return 0.0
    layer = _ATM[0]
    for entry in _ATM:
        if alt_km >= entry[0]:
            layer = entry
        else:
            break
    base_alt, H, rho0 = layer
    return rho0 * math.exp(-(alt_km - base_alt) / H) * 1e9   # → kg/km³


# ═══════════════════════════════════════════════════════════════════════════
# Result data-classes
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PropagationResult:
    """
    Output of a single :func:`propagate_state` call.

    Attributes
    ----------
    state : StateVector
        Updated ECI state after advancing by *dt* seconds.
    dt : float
        Integration step actually used [s].
    altitude_km : float
        Scalar altitude above mean Earth surface [km].
    speed_km_s : float
        Scalar speed [km/s].
    reentry : bool
        True if the object's altitude dropped below 0 km (re-entry detected).
    accelerations : dict
        Component accelerations [km/s²] for diagnostics/logging.
    """
    state:         StateVector
    dt:            float
    altitude_km:   float
    speed_km_s:    float
    reentry:       bool
    accelerations: dict = field(default_factory=dict)


@dataclass
class OrbitSimulationResult:
    """
    Output of :func:`simulate_orbit`.

    Attributes
    ----------
    object_id : str
    object_type : str  – "satellite" or "debris"
    epoch_start : datetime
    epoch_end : datetime
    steps : int  – number of integration steps taken
    trajectory : list[StateVector]  – one snapshot per step
    altitudes_km : list[float]
    speeds_km_s : list[float]
    final_state : StateVector  – last computed state
    reentry_detected : bool
    reentry_time_s : float | None  – elapsed seconds at re-entry (or None)
    """
    object_id:        str
    object_type:      str
    epoch_start:      datetime
    epoch_end:        datetime
    steps:            int
    trajectory:       List[StateVector]
    altitudes_km:     List[float]
    speeds_km_s:      List[float]
    final_state:      StateVector
    reentry_detected: bool
    reentry_time_s:   Optional[float]

    def to_dict(self) -> dict:
        """Serialise result to a JSON-safe dict (trajectory as list of dicts)."""
        return {
            "object_id":        self.object_id,
            "object_type":      self.object_type,
            "epoch_start":      self.epoch_start.isoformat(),
            "epoch_end":        self.epoch_end.isoformat(),
            "steps":            self.steps,
            "reentry_detected": self.reentry_detected,
            "reentry_time_s":   self.reentry_time_s,
            "final_state": {
                "x":  self.final_state.x,  "y":  self.final_state.y,  "z":  self.final_state.z,
                "vx": self.final_state.vx, "vy": self.final_state.vy, "vz": self.final_state.vz,
            },
            "trajectory": [
                {"x": s.x, "y": s.y, "z": s.z,
                 "vx": s.vx, "vy": s.vy, "vz": s.vz}
                for s in self.trajectory
            ],
        }


# ═══════════════════════════════════════════════════════════════════════════
# Core physics helpers
# ═══════════════════════════════════════════════════════════════════════════

def _gravitational_acceleration(r: np.ndarray) -> np.ndarray:
    """
    Two-body (point-mass) gravitational acceleration.

    .. math::
        \\mathbf{a}_{2B} = -\\frac{\\mu}{r^3} \\mathbf{r}

    Parameters
    ----------
    r : ndarray, shape (3,)
        ECI position vector [km].

    Returns
    -------
    ndarray, shape (3,)
        Acceleration [km/s²].
    """
    r_norm = np.linalg.norm(r)
    return -(MU / r_norm ** 3) * r


def _j2_acceleration(r: np.ndarray) -> np.ndarray:
    """
    J2 oblateness perturbation acceleration.

    The J2 term captures the dominant effect of Earth's equatorial bulge on
    orbital mechanics.  In ECI components:

    .. math::
        a_{J2,x} = -\\frac{3}{2} \\frac{J_2 \\mu R_e^2}{r^5}
                    x \\left(1 - 5\\frac{z^2}{r^2}\\right)

        a_{J2,y} = -\\frac{3}{2} \\frac{J_2 \\mu R_e^2}{r^5}
                    y \\left(1 - 5\\frac{z^2}{r^2}\\right)

        a_{J2,z} = -\\frac{3}{2} \\frac{J_2 \\mu R_e^2}{r^5}
                    z \\left(3 - 5\\frac{z^2}{r^2}\\right)

    Reference:  Vallado, D. A. – "Fundamentals of Astrodynamics and
                Applications", 4th ed., §9.6.

    Parameters
    ----------
    r : ndarray, shape (3,)
        ECI position vector [km].

    Returns
    -------
    ndarray, shape (3,)
        J2 acceleration [km/s²].
    """
    r_norm  = np.linalg.norm(r)
    r2      = r_norm * r_norm
    z2_over_r2 = (r[2] / r_norm) ** 2

    factor = -1.5 * J2 * MU * RE_EQ ** 2 / (r2 ** 2 * r_norm)

    ax = factor * r[0] * (1.0 - 5.0 * z2_over_r2)
    ay = factor * r[1] * (1.0 - 5.0 * z2_over_r2)
    az = factor * r[2] * (3.0 - 5.0 * z2_over_r2)

    return np.array([ax, ay, az])


def _drag_acceleration(
    r: np.ndarray,
    v: np.ndarray,
    Cd:  float = 2.2,
    A_m: float = 0.01,       # cross-section / mass  [km² / kg  = m² / kg × 1e-6]
) -> np.ndarray:
    """
    Atmospheric drag deceleration.

    .. math::
        \\mathbf{a}_{drag} = -\\frac{1}{2} C_D \\frac{A}{m} \\rho v_{rel}^2
                              \\hat{v}_{rel}

    For LEO, Earth's atmosphere co-rotates with the surface; the relative
    velocity includes the rotational term.

    Parameters
    ----------
    r    : ndarray (3,)  – ECI position [km]
    v    : ndarray (3,)  – ECI velocity [km/s]
    Cd   : float         – drag coefficient (dimensionless, default 2.2)
    A_m  : float         – ballistic coefficient area/mass [m²/kg]
                           Converted internally: 1 m² = 1e-6 km²

    Returns
    -------
    ndarray (3,)  – drag acceleration [km/s²]
    """
    alt_km = np.linalg.norm(r) - RE
    rho    = _atm_density_km3(alt_km)          # kg/km³
    if rho == 0.0:
        return np.zeros(3)

    # Earth rotation vector  ω_⊕ × r  [km/s]
    OMEGA_EARTH = 7.2921150e-5   # rad/s
    v_earth = np.array([-OMEGA_EARTH * r[1], OMEGA_EARTH * r[0], 0.0])
    v_rel   = v - v_earth
    v_rel_norm = np.linalg.norm(v_rel)
    if v_rel_norm < 1e-10:
        return np.zeros(3)

    # A_m in m²/kg → km²/kg: multiply by 1e-6
    A_m_km2 = A_m * 1e-6
    return -0.5 * Cd * A_m_km2 * rho * v_rel_norm * v_rel


def _total_acceleration(
    r: np.ndarray,
    v: np.ndarray,
    include_j2:   bool  = True,
    include_drag: bool  = True,
    Cd:  float = 2.2,
    A_m: float = 0.01,
) -> tuple[np.ndarray, dict]:
    """
    Compute total ECI acceleration and return individual components.

    Returns
    -------
    a_total : ndarray (3,)
    components : dict  { "gravity_km_s2", "j2_km_s2", "drag_km_s2" }
    """
    a_grav = _gravitational_acceleration(r)
    a_j2   = _j2_acceleration(r)   if include_j2   else np.zeros(3)
    a_drag = _drag_acceleration(r, v, Cd, A_m) if include_drag else np.zeros(3)
    a_total = a_grav + a_j2 + a_drag
    components = {
        "gravity_km_s2": float(np.linalg.norm(a_grav)),
        "j2_km_s2":      float(np.linalg.norm(a_j2)),
        "drag_km_s2":    float(np.linalg.norm(a_drag)),
    }
    return a_total, components


# ═══════════════════════════════════════════════════════════════════════════
# RK4 integrator
# ═══════════════════════════════════════════════════════════════════════════

def _ode(
    state6: np.ndarray,
    include_j2:   bool,
    include_drag: bool,
    Cd:  float,
    A_m: float,
) -> np.ndarray:
    """
    Right-hand side of the equations of motion.

    State vector layout:  [x, y, z, vx, vy, vz]
    Derivative layout:    [vx, vy, vz, ax, ay, az]
    """
    r = state6[:3]
    v = state6[3:]
    a, _ = _total_acceleration(r, v, include_j2, include_drag, Cd, A_m)
    return np.concatenate([v, a])


def _rk4_step(
    state6: np.ndarray,
    dt: float,
    include_j2:   bool,
    include_drag: bool,
    Cd:  float,
    A_m: float,
) -> np.ndarray:
    """
    Advance state by one RK4 step of size *dt*.

    The classic 4-stage Runge-Kutta scheme:

        k1 = f(y)
        k2 = f(y + h/2 · k1)
        k3 = f(y + h/2 · k2)
        k4 = f(y + h   · k3)
        y_new = y + h/6 · (k1 + 2k2 + 2k3 + k4)

    Local truncation error is O(h⁵); global error is O(h⁴).
    """
    args = (include_j2, include_drag, Cd, A_m)
    k1 = _ode(state6,               *args)
    k2 = _ode(state6 + 0.5*dt * k1, *args)
    k3 = _ode(state6 + 0.5*dt * k2, *args)
    k4 = _ode(state6 +     dt * k3, *args)
    return state6 + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)


# ═══════════════════════════════════════════════════════════════════════════
# Public function 1 – propagate_state
# ═══════════════════════════════════════════════════════════════════════════

def propagate_state(
    state: StateVector,
    dt: float,
    *,
    include_j2:    bool  = True,
    include_drag:  bool  = True,
    drag_coeff:    float = 2.2,
    area_m2:       float = 2.0,
    mass_kg:       float = 100.0,
) -> PropagationResult:
    """
    Advance a single ECI state vector by *dt* seconds using RK4.

    This is the **lowest-level building block** of the propagation engine.
    Use :func:`simulate_orbit` for multi-step full-orbit simulations.

    Parameters
    ----------
    state : StateVector
        Current ECI state (x, y, z [km]; vx, vy, vz [km/s]).
    dt : float
        Integration time step [s].  Negative values integrate backward.
    include_j2 : bool
        Apply J2 oblateness perturbation (default True).
    include_drag : bool
        Apply atmospheric drag perturbation (default True).
    drag_coeff : float
        Drag coefficient C_D (dimensionless).
    area_m2 : float
        Mean cross-sectional area [m²].
    mass_kg : float
        Object mass [kg].

    Returns
    -------
    PropagationResult
        Updated StateVector plus diagnostics.

    Raises
    ------
    ValueError
        If *dt* is zero or if *state* has a sub-surface position.

    Examples
    --------
    >>> from app.models.sim_objects import StateVector
    >>> from app.physics.orbit_propagator import propagate_state
    >>> sv = StateVector(x=6778.0, y=0.0, z=0.0, vx=0.0, vy=7.669, vz=0.0)
    >>> result = propagate_state(sv, dt=60.0)
    >>> round(result.altitude_km, 1)
    407.0
    """
    if dt == 0.0:
        raise ValueError("dt must be non-zero")

    A_m = area_m2 / mass_kg    # ballistic area-to-mass ratio  [m²/kg]

    state6 = np.array([state.x, state.y, state.z, state.vx, state.vy, state.vz])
    new_state6 = _rk4_step(state6, dt, include_j2, include_drag, drag_coeff, A_m)

    r_new = new_state6[:3]
    v_new = new_state6[3:]

    alt   = float(np.linalg.norm(r_new)) - RE
    speed = float(np.linalg.norm(v_new))

    # Capture acceleration components at the NEW position for diagnostics
    _, components = _total_acceleration(
        r_new, v_new, include_j2, include_drag, drag_coeff, A_m
    )

    new_sv = StateVector(
        x=float(r_new[0]), y=float(r_new[1]), z=float(r_new[2]),
        vx=float(v_new[0]), vy=float(v_new[1]), vz=float(v_new[2]),
    )

    return PropagationResult(
        state        = new_sv,
        dt           = dt,
        altitude_km  = alt,
        speed_km_s   = speed,
        reentry      = alt < 0.0,
        accelerations = components,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Public function 2 – simulate_orbit
# ═══════════════════════════════════════════════════════════════════════════

def simulate_orbit(
    obj: Union[SimSatellite, SimDebris],
    time_seconds: float,
    *,
    step_seconds:  float = 60.0,
    include_j2:    bool  = True,
    include_drag:  bool  = True,
    drag_coeff:    float = 2.2,
    area_m2:       float = 2.0,
    mass_kg:       Optional[float] = None,
    epoch:         Optional[datetime] = None,
    record_every:  int   = 1,
) -> OrbitSimulationResult:
    """
    Propagate a :class:`SimSatellite` or :class:`SimDebris` object for a
    total duration of *time_seconds*, collecting trajectory snapshots.

    The integrator advances the ECI state in fixed steps of *step_seconds*
    using the :func:`propagate_state` RK4 engine.  Propagation halts early
    if the object's altitude drops below 0 km (re-entry).

    Parameters
    ----------
    obj : SimSatellite | SimDebris
        Object to propagate.  Its current ``position`` and ``velocity``
        fields are used as the initial conditions.
    time_seconds : float
        Total propagation duration [s].  Must be positive.
    step_seconds : float
        RK4 integration step size [s] (default 60 s).
        Smaller values → higher accuracy, more snapshots, slower runtime.
    include_j2 : bool
        Apply J2 perturbation (default True).
    include_drag : bool
        Apply atmospheric drag (default True).
    drag_coeff : float
        Drag coefficient C_D.  Ignored when *include_drag* is False.
    area_m2 : float
        Cross-sectional area [m²] for drag calculation.
        If *obj* is a ``SimSatellite`` and you don't override this, the
        default 2.0 m² is used.
    mass_kg : float | None
        Object mass [kg].  Defaults to ``obj.fuel_kg + 100.0`` for
        satellites (crude dry-mass estimate) or 10.0 kg for debris.
    epoch : datetime | None
        Simulation start epoch (UTC).  Defaults to ``datetime.utcnow()``.
    record_every : int
        Record a trajectory snapshot every N steps (down-sample factor).
        Useful for long simulations with many steps.

    Returns
    -------
    OrbitSimulationResult
        Full trajectory, altitude/speed series, and re-entry info.

    Raises
    ------
    ValueError
        If *time_seconds* ≤ 0 or *step_seconds* ≤ 0.

    Examples
    --------
    >>> from app.models.sim_objects import SimSatellite, StateVector
    >>> from app.physics.orbit_propagator import simulate_orbit
    >>> sat = SimSatellite(
    ...     name='ISS',
    ...     position=StateVector(x=6778.0, y=0.0, z=0.0, vx=0.0, vy=0.0, vz=0.0),
    ...     velocity=StateVector(x=0.0, y=0.0, z=0.0, vx=0.0, vy=7.669, vz=0.0),
    ...     fuel_kg=120.0,
    ... )
    >>> result = simulate_orbit(sat, time_seconds=5400.0, step_seconds=60.0)
    >>> result.steps
    90
    """
    if time_seconds <= 0:
        raise ValueError(f"time_seconds must be positive, got {time_seconds}")
    if step_seconds <= 0:
        raise ValueError(f"step_seconds must be positive, got {step_seconds}")

    # ── Determine mass ───────────────────────────────────────────────────────
    if mass_kg is None:
        if isinstance(obj, SimSatellite):
            mass_kg = max(1.0, obj.fuel_kg + 100.0)   # rough dry-mass estimate
        else:
            mass_kg = 10.0

    # ── Initial state ────────────────────────────────────────────────────────
    is_satellite  = isinstance(obj, SimSatellite)
    object_type   = "satellite" if is_satellite else "debris"
    epoch_start   = epoch or datetime.utcnow()

    # Read position and velocity from the model's two StateVector fields
    current_sv = StateVector(
        x=obj.position.x, y=obj.position.y, z=obj.position.z,
        vx=obj.velocity.vx, vy=obj.velocity.vy, vz=obj.velocity.vz,
    )

    trajectory:   List[StateVector] = []
    altitudes:    List[float]       = []
    speeds:       List[float]       = []

    elapsed        = 0.0
    step_count     = 0
    reentry        = False
    reentry_time_s: Optional[float] = None

    logger.info(
        "simulate_orbit START  id=%s  type=%s  duration=%.0f s  step=%.0f s",
        obj.id, object_type, time_seconds, step_seconds,
    )

    while elapsed < time_seconds:
        # Clamp last step so we don't overshoot
        dt = min(step_seconds, time_seconds - elapsed)

        result = propagate_state(
            current_sv, dt,
            include_j2   = include_j2,
            include_drag = include_drag,
            drag_coeff   = drag_coeff,
            area_m2      = area_m2,
            mass_kg      = mass_kg,
        )

        current_sv = result.state
        step_count += 1
        elapsed    += dt

        # Record snapshot (honour down-sample factor)
        if step_count % record_every == 0 or result.reentry:
            trajectory.append(current_sv)
            altitudes.append(result.altitude_km)
            speeds.append(result.speed_km_s)

        if result.reentry:
            reentry        = True
            reentry_time_s = elapsed
            logger.warning(
                "Re-entry detected  id=%s  t=%.1f s  alt=%.2f km",
                obj.id, elapsed, result.altitude_km,
            )
            break

    epoch_end = epoch_start + timedelta(seconds=elapsed)

    logger.info(
        "simulate_orbit END  id=%s  steps=%d  elapsed=%.1f s  reentry=%s",
        obj.id, step_count, elapsed, reentry,
    )

    return OrbitSimulationResult(
        object_id        = obj.id,
        object_type      = object_type,
        epoch_start      = epoch_start,
        epoch_end        = epoch_end,
        steps            = step_count,
        trajectory       = trajectory,
        altitudes_km     = altitudes,
        speeds_km_s      = speeds,
        final_state      = current_sv,
        reentry_detected = reentry,
        reentry_time_s   = reentry_time_s,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Convenience helpers
# ═══════════════════════════════════════════════════════════════════════════

def orbital_period_s(semi_major_axis_km: float) -> float:
    """
    Kepler's third law: period T = 2π √(a³/μ)  [seconds].

    Parameters
    ----------
    semi_major_axis_km : float
        Orbit semi-major axis [km].
    """
    return 2.0 * math.pi * math.sqrt(semi_major_axis_km ** 3 / MU)


def circular_velocity_km_s(altitude_km: float) -> float:
    """
    Circular orbit velocity at a given altitude: v = √(μ / (Rₑ + h)).
    """
    return math.sqrt(MU / (RE + altitude_km))


def state_from_altitude(
    altitude_km: float,
    inclination_deg: float = 0.0,
) -> StateVector:
    """
    Build a circular-orbit StateVector in the ECI equatorial plane
    at the given altitude.  Useful for quick test setup.

    Parameters
    ----------
    altitude_km : float
        Orbit altitude above mean Earth surface [km].
    inclination_deg : float
        Orbit inclination [°].  0 = equatorial.
    """
    r = RE + altitude_km
    v = circular_velocity_km_s(altitude_km)
    inc = math.radians(inclination_deg)
    return StateVector(
        x=r,  y=0.0, z=0.0,
        vx=0.0,
        vy=v * math.cos(inc),
        vz=v * math.sin(inc),
    )
