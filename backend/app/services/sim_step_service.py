"""
sim_step_service.py
===================
Discrete-time simulation stepping engine for the Autonomous Constellation Manager.

One call to :func:`SimStepService.step` advances the **entire registry** by
``step_seconds`` seconds:

    Phase 1 – Propagation
        RK4-propagate every satellite and every debris object in the registry
        by exactly ``step_seconds`` seconds.  Updated state is committed back
        to the registry via ``fast_update_satellite / fast_update_debris``.

    Phase 2 – Maneuver execution
        Scan the ``schedule_store`` for any burns due within the current time
        window  [sim_clock, sim_clock + step_seconds).  For each due burn:
          • Re-validate fuel with the Tsiolkovsky equation.
          • Commit the velocity impulse (ΔV) to the satellite state.
          • Deduct propellant via ``fuel_calculator.apply_burn_to_registry``.
          • Mark the ScheduledBurn as EXECUTED.

    Phase 3 – Collision detection
        Build a KDTree from the post-propagation debris positions and run a
        ball-query for every satellite.  Close approaches within
        ``collision_threshold_km`` are counted and returned as collision events.

    Phase 4 – Clock advance
        Advance the module-level simulation clock by ``step_seconds``.

All three phases are synchronous.  For large constellations the step may take
O(100 ms); callers should use a background task if needed.

Simulation clock
----------------
The service maintains a **sim_clock** (wall-clock naive UTC datetime) that
starts at the moment the first step is called.  It can be reset via
:meth:`reset_clock`.  The clock is used to:
  • Determine which scheduled burns are due.
  • Attach ``new_timestamp`` to the step response.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.physics.orbit_propagator import propagate_state
from app.physics.fuel_calculator import (
    propellant_mass, is_feasible,
    apply_burn_to_registry, InsufficientFuelError,
    ISP_S, MIN_FUEL_KG,
)
from app.models.sim_objects import StateVector

logger = logging.getLogger(__name__)

# ── Default physics parameters for unregistered (debris) objects ──────────────
_DEBRIS_MASS_KG:  float = 10.0    # kg  — default debris mass
_DEBRIS_AREA_M2:  float = 0.1     # m²  — default debris cross-section
_SAT_MASS_KG:     float = 100.0   # kg  — default satellite dry mass
_SAT_AREA_M2:     float = 2.0     # m²  — default satellite cross-section
_SAT_DRY_KG:      float = 100.0   # kg  — dry mass for Tsiolkovsky

COLLISION_THRESHOLD_KM: float = 0.1   # 100 m — below this distance = collision event


# ═══════════════════════════════════════════════════════════════════════════
# Output data-classes
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CollisionEvent:
    """A close-approach event detected during collision screening."""
    satellite_id:  str
    debris_id:     str
    distance_km:   float
    satellite_pos: Tuple[float, float, float]
    debris_pos:    Tuple[float, float, float]

    def to_dict(self) -> dict:
        return {
            "satellite_id": self.satellite_id,
            "debris_id":    self.debris_id,
            "distance_km":  round(self.distance_km, 6),
            "satellite_pos_km": list(self.satellite_pos),
            "debris_pos_km":    list(self.debris_pos),
        }


@dataclass
class ManeuverExecution:
    """Record of a maneuver executed during this step."""
    burn_id:        str
    satellite_id:   str
    delta_v_mag_m_s: float
    propellant_used_kg: float
    fuel_before:    float
    fuel_after:     float
    success:        bool
    reason:         Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "burn_id":             self.burn_id,
            "satellite_id":        self.satellite_id,
            "delta_v_mag_m_s":     round(self.delta_v_mag_m_s, 4),
            "propellant_used_kg":  round(self.propellant_used_kg, 6),
            "fuel_before_kg":      round(self.fuel_before, 4),
            "fuel_after_kg":       round(self.fuel_after, 4),
            "success":             self.success,
            "reason":              self.reason,
        }


@dataclass
class StepResult:
    """Full result from one simulation step."""
    status:              str = "STEP_COMPLETE"
    step_seconds:        float = 0.0
    sim_clock_before:    str = ""
    new_timestamp:       str = ""
    satellites_propagated: int = 0
    debris_propagated:   int = 0
    reentries_detected:  int = 0
    maneuvers_executed:  int = 0
    maneuvers_attempted: int = 0
    collisions_detected: int = 0
    collision_events:    List[CollisionEvent] = field(default_factory=list)
    maneuver_log:        List[ManeuverExecution] = field(default_factory=list)
    wall_time_ms:        float = 0.0

    def to_dict(self) -> dict:
        return {
            "status":                self.status,
            "step_seconds":          self.step_seconds,
            "sim_clock_before":      self.sim_clock_before,
            "new_timestamp":         self.new_timestamp,
            "satellites_propagated": self.satellites_propagated,
            "debris_propagated":     self.debris_propagated,
            "reentries_detected":    self.reentries_detected,
            "maneuvers_executed":    self.maneuvers_executed,
            "maneuvers_attempted":   self.maneuvers_attempted,
            "collisions_detected":   self.collisions_detected,
            "collision_events":      [e.to_dict() for e in self.collision_events],
            "maneuver_log":          [m.to_dict() for m in self.maneuver_log],
            "wall_time_ms":          round(self.wall_time_ms, 2),
        }


# ═══════════════════════════════════════════════════════════════════════════
# Simulation Step Service
# ═══════════════════════════════════════════════════════════════════════════

class SimStepService:
    """
    Discrete-time simulation stepper.

    Parameters
    ----------
    collision_threshold_km : float
        Miss-distance threshold for flagging a collision event [km].
    include_j2 : bool
        Apply J2 oblateness in propagation.
    include_drag : bool
        Apply atmospheric drag in propagation.
    """

    def __init__(
        self,
        collision_threshold_km: float = COLLISION_THRESHOLD_KM,
        include_j2:   bool = True,
        include_drag: bool = True,
    ) -> None:
        self.collision_threshold_km = collision_threshold_km
        self.include_j2   = include_j2
        self.include_drag = include_drag
        self._sim_clock:  Optional[datetime] = None
        self._step_count: int = 0

    # ── public interface ──────────────────────────────────────────────────────

    @property
    def sim_clock(self) -> datetime:
        if self._sim_clock is None:
            self._sim_clock = datetime.utcnow()
        return self._sim_clock

    def reset_clock(self, epoch: Optional[datetime] = None) -> None:
        """Reset simulation clock to *epoch* (default: now)."""
        self._sim_clock = epoch or datetime.utcnow()
        self._step_count = 0
        logger.info("Simulation clock reset to %s", self._sim_clock.isoformat())

    def step(self, step_seconds: float) -> StepResult:
        """
        Advance the simulation by *step_seconds*.

        Executes the four-phase pipeline:
          1. Propagate (RK4) → update registry
          2. Execute due scheduled maneuvers
          3. Detect collisions (KDTree)
          4. Advance clock

        Parameters
        ----------
        step_seconds : float – step duration [s], must be > 0

        Returns
        -------
        StepResult
        """
        if step_seconds <= 0:
            raise ValueError(f"step_seconds must be positive, got {step_seconds}")

        t0 = time.perf_counter()
        result = StepResult(step_seconds=step_seconds)
        result.sim_clock_before = self.sim_clock.isoformat()

        # ── Phase 1: Propagation ──────────────────────────────────────────────
        self._phase_propagate(step_seconds, result)

        # ── Phase 2: Maneuver execution ───────────────────────────────────────
        self._phase_execute_maneuvers(step_seconds, result)

        # ── Phase 3: Collision detection ──────────────────────────────────────
        self._phase_detect_collisions(result)

        # ── Phase 4: Advance clock ────────────────────────────────────────────
        self._sim_clock = self.sim_clock + timedelta(seconds=step_seconds)
        self._step_count += 1
        result.new_timestamp = self._sim_clock.isoformat()

        result.wall_time_ms = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "STEP #%d complete  dt=%.0f s  sats=%d  debris=%d  "
            "maneuvers=%d  collisions=%d  wall=%.1f ms",
            self._step_count, step_seconds,
            result.satellites_propagated, result.debris_propagated,
            result.maneuvers_executed, result.collisions_detected,
            result.wall_time_ms,
        )
        return result

    # ── Phase 1: propagate all objects ────────────────────────────────────────

    def _phase_propagate(self, dt: float, result: StepResult) -> None:
        """RK4-propagate every registry object and commit state back."""
        from app.data.registry import satellites, debris, fast_update_satellite, fast_update_debris

        # Propagate satellites
        for sat_id, sat in list(satellites.items()):
            state = StateVector(
                x=sat.position.x, y=sat.position.y, z=sat.position.z,
                vx=sat.velocity.vx, vy=sat.velocity.vy, vz=sat.velocity.vz,
            )
            mass = max(1.0, sat.fuel_kg + _SAT_DRY_KG)
            try:
                prop = propagate_state(
                    state, dt,
                    include_j2   = self.include_j2,
                    include_drag = self.include_drag,
                    area_m2      = _SAT_AREA_M2,
                    mass_kg      = mass,
                )
                ns = prop.state
                fast_update_satellite(sat_id, ns.x, ns.y, ns.z, ns.vx, ns.vy, ns.vz)
                result.satellites_propagated += 1
                if prop.reentry:
                    result.reentries_detected += 1
                    logger.warning("Re-entry: satellite %s at dt=%.0f s", sat_id, dt)
            except Exception as exc:
                logger.error("Propagation failed for sat %s: %s", sat_id, exc)

        # Propagate debris
        for deb_id, deb in list(debris.items()):
            state = StateVector(
                x=deb.position.x, y=deb.position.y, z=deb.position.z,
                vx=deb.velocity.vx, vy=deb.velocity.vy, vz=deb.velocity.vz,
            )
            try:
                prop = propagate_state(
                    state, dt,
                    include_j2   = self.include_j2,
                    include_drag = self.include_drag,
                    area_m2      = _DEBRIS_AREA_M2,
                    mass_kg      = _DEBRIS_MASS_KG,
                )
                ns = prop.state
                fast_update_debris(deb_id, ns.x, ns.y, ns.z, ns.vx, ns.vy, ns.vz)
                result.debris_propagated += 1
                if prop.reentry:
                    result.reentries_detected += 1
            except Exception as exc:
                logger.error("Propagation failed for debris %s: %s", deb_id, exc)

    # ── Phase 2: execute due scheduled maneuvers ──────────────────────────────

    def _phase_execute_maneuvers(self, dt: float, result: StepResult) -> None:
        """
        Find ScheduledBurns whose burn_time falls in [sim_clock, sim_clock + dt)
        and commit them to the registry.
        """
        from app.services.maneuver_scheduler import schedule_store
        from app.data.registry import get_satellite, fast_update_satellite

        window_start = self.sim_clock
        window_end   = self.sim_clock + timedelta(seconds=dt)

        for satellite_id, burn_list in schedule_store.items():
            for sb in burn_list:
                if sb.status != "SCHEDULED":
                    continue
                if not (window_start <= sb.burn_time < window_end):
                    continue

                result.maneuvers_attempted += 1
                dv_mag = sb.delta_v_mag_m_s
                burn_log = ManeuverExecution(
                    burn_id         = sb.burn_id,
                    satellite_id    = satellite_id,
                    delta_v_mag_m_s = dv_mag,
                    propellant_used_kg = 0.0,
                    fuel_before     = 0.0,
                    fuel_after      = 0.0,
                    success         = False,
                )

                sat = get_satellite(satellite_id)
                if sat is None:
                    burn_log.reason = f"Satellite '{satellite_id}' not in registry at burn time"
                    sb.status = "FAILED"
                    result.maneuver_log.append(burn_log)
                    continue

                burn_log.fuel_before = sat.fuel_kg

                try:
                    # Commit fuel deduction (Tsiolkovsky, Isp=300 s)
                    record = apply_burn_to_registry(
                        satellite_id = satellite_id,
                        burn_id      = sb.burn_id,
                        delta_v_m_s  = dv_mag,
                        dry_mass_kg  = _SAT_DRY_KG,
                        isp_s        = ISP_S,
                    )

                    # Apply ΔV impulse to satellite velocity
                    dv  = sb.delta_v_vector         # {"x", "y", "z"} in km/s
                    pos = sat.position
                    vel = sat.velocity
                    fast_update_satellite(
                        satellite_id,
                        pos.x, pos.y, pos.z,
                        vel.vx + float(dv.get("x", 0.0)),
                        vel.vy + float(dv.get("y", 0.0)),
                        vel.vz + float(dv.get("z", 0.0)),
                    )

                    sb.status               = "EXECUTED"
                    burn_log.success        = True
                    burn_log.fuel_after     = record.fuel_after
                    burn_log.propellant_used_kg = record.propellant_used
                    result.maneuvers_executed += 1
                    logger.info(
                        "MANEUVER EXECUTED  sat=%s  burn=%s  ΔV=%.3f m/s  fuel: %.3f→%.3f kg",
                        satellite_id, sb.burn_id, dv_mag,
                        burn_log.fuel_before, burn_log.fuel_after,
                    )

                except InsufficientFuelError as e:
                    sb.status      = "FAILED"
                    burn_log.reason = str(e)
                    burn_log.fuel_after = sat.fuel_kg
                    logger.warning("Burn FAILED (fuel): sat=%s burn=%s – %s", satellite_id, sb.burn_id, e)

                except Exception as exc:
                    sb.status      = "FAILED"
                    burn_log.reason = str(exc)
                    burn_log.fuel_after = sat.fuel_kg
                    logger.error("Burn FAILED: sat=%s burn=%s – %s", satellite_id, sb.burn_id, exc)

                result.maneuver_log.append(burn_log)

    # ── Phase 3: collision detection ──────────────────────────────────────────

    def _phase_detect_collisions(self, result: StepResult) -> None:
        """
        Build a KDTree from post-propagation debris positions and screen
        all satellites for close approaches ≤ collision_threshold_km.
        """
        from app.data.registry import satellites, debris

        if not satellites or not debris:
            return

        deb_list  = list(debris.values())
        deb_ids   = [d.id for d in deb_list]
        deb_pos   = np.array([[d.position.x, d.position.y, d.position.z] for d in deb_list])

        if len(deb_pos) == 0:
            return

        try:
            from scipy.spatial import KDTree
            tree = KDTree(deb_pos)
        except Exception as exc:
            logger.error("KDTree build failed: %s", exc)
            return

        for sat_id, sat in satellites.items():
            sat_pos = np.array([sat.position.x, sat.position.y, sat.position.z])
            indices = tree.query_ball_point(sat_pos, self.collision_threshold_km)
            for idx in indices:
                dist = float(np.linalg.norm(sat_pos - deb_pos[idx]))
                event = CollisionEvent(
                    satellite_id = sat_id,
                    debris_id    = deb_ids[idx],
                    distance_km  = dist,
                    satellite_pos = (sat.position.x, sat.position.y, sat.position.z),
                    debris_pos    = (deb_list[idx].position.x,
                                     deb_list[idx].position.y,
                                     deb_list[idx].position.z),
                )
                result.collision_events.append(event)
                result.collisions_detected += 1
                logger.critical(
                    "COLLISION EVENT  sat=%s  debris=%s  dist=%.4f km",
                    sat_id, deb_ids[idx], dist,
                )


# ═══════════════════════════════════════════════════════════════════════════
# Module-level singleton
# ═══════════════════════════════════════════════════════════════════════════

_step_service: Optional[SimStepService] = None


def get_step_service() -> SimStepService:
    global _step_service
    if _step_service is None:
        _step_service = SimStepService()
    return _step_service


def reset_step_service() -> None:
    global _step_service
    _step_service = None
