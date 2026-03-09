"""
avoidance_planner.py
====================
Autonomous Collision Avoidance Planner for the Autonomous Constellation Manager.

Overview
--------
For every high-risk conjunction event (sourced from ConjunctionDetectionService),
the planner computes a minimum-fuel maneuver that increases the miss distance at
TCA above the safe-separation threshold.

Coordinate frames
-----------------
All computations use the ECI J2000 frame internally, but maneuver vectors are
also decomposed into the Radial–Transverse–Normal (RTN) Orbital Frame for easier
operational interpretation:

  R̂  (Radial)     – points from Earth centre outward through the satellite
  T̂  (Transverse) – along-track, in the direction of orbital motion (prograde)
  N̂  (Normal)     – normal to the orbital plane (cross-track)

Why RTN matters
---------------
For a close approach where the debris passes near the satellite in the
along-track direction (the most common LEO geometry), a **transverse (T)**
impulse efficiently changes orbital phasing — moving the satellite earlier
or later in its orbit so the threat passes through empty space.  This is
the lowest-energy avoidance strategy.

For a radially-close approach, a **radial (R)** burn changes the orbit
shape to offset the flyby altitude.

Algorithm (per conjunction)
----------------------------
1. Compute RTN unit vectors at the satellite's current position / velocity.
2. Project the miss-vector at TCA (debris_pos_TCA − sat_pos_TCA) onto RTN.
3. Determine the avoidance direction:
   a. If |T component| > |R component|: maneuver primarily in ±T (prograde).
   b. Otherwise: maneuver primarily in ±R (radial).
4. The sign is chosen to push the satellite AWAY from the debris at TCA.
5. Scale to the minimum ΔV that achieves safe separation; cap at MAX_DV_KM_S.
6. Convert back to ECI for the burn vector output.

Fuel and cooldown constraints
------------------------------
  MAX_DV_KM_S   = 0.015 km/s = 15 m/s   hard ΔV cap per burn
  COOLDOWN_S    = 600 s                  minimum gap between satellite burns
  MIN_FUEL_KG   = 1.0 kg                 satellite must have at least this much fuel

Output schema
-------------
Each burn is returned as:
  {
    "burn_id":       str (UUID),
    "satellite_id":  str,
    "debris_id":     str,
    "burn_time":     str (ISO8601 UTC – when to execute the burn),
    "delta_v_km_s":  {"x": float, "y": float, "z": float},
    "delta_v_m_s":   {"x": float, "y": float, "z": float},
    "delta_v_mag_m_s": float,
    "burn_frame":    {"R": float, "T": float, "N": float},
    "maneuver_type": "PROGRADE"|"RETROGRADE"|"RADIAL_PLUS"|"RADIAL_MINUS",
    "fuel_cost_kg":  float,
    "risk_level_before": str,
    "estimated_miss_after_km": float,
    "feasible":      bool,
    "infeasible_reason": str | None,
  }
"""

from __future__ import annotations

import logging
import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.services.conjunction_service import (
    ConjunctionDataMessage,
    RISK_CRITICAL_KM,
    RISK_WARNING_KM,
    RISK_CAUTION_KM,
)

from app.physics.fuel_calculator import (
    propellant_mass,
    mass_after_burn,
    max_delta_v_m_s,
    is_feasible as _fuel_is_feasible,
    apply_burn_to_registry,
    fuel_budget_summary,
    InsufficientFuelError,
    ISP_S,
    MIN_FUEL_KG,
    FUEL_RESERVE_KG,
)

logger = logging.getLogger(__name__)

# ── Constraint constants ─────────────────────────────────────────────────────
MAX_DV_KM_S:      float = 0.015      # 15 m/s hard cap per burn
COOLDOWN_S:       float = 600.0      # minimum seconds between burns per satellite
SAFE_SEP_KM:      float = 2.0        # target safe miss distance after maneuver
BURN_LEAD_TIME_S: float = 300.0      # execute burn 5 min before TCA by default
SAT_DRY_MASS_KG:  float = 100.0      # default dry-mass estimate for Tsiolkovsky


# ═══════════════════════════════════════════════════════════════════════════
# Result data-class
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ManeuverPlan:
    """
    A single collision avoidance burn recommendation.

    All vector quantities are in ECI [km/s] unless noted.
    """
    burn_id:                str
    satellite_id:           str
    satellite_name:         str
    debris_id:              str
    debris_designation:     str

    burn_time:              str          # ISO8601 UTC
    burn_time_offset_s:     float        # seconds from epoch until burn
    lead_time_to_tca_s:     float        # how far before TCA the burn fires

    delta_v_km_s:           dict         # {"x", "y", "z"}
    delta_v_m_s:            dict         # {"x", "y", "z"}  (for human readability)
    delta_v_mag_m_s:        float        # |ΔV| in m/s
    burn_frame:             dict         # {"R", "T", "N"} components [m/s]
    maneuver_type:          str          # PROGRADE | RETROGRADE | RADIAL_PLUS | RADIAL_MINUS

    fuel_cost_kg:           float        # propellant mass consumed [kg]
    risk_level_before:      str
    estimated_miss_after_km: float       # coarse linear estimate of new miss distance

    feasible:               bool
    infeasible_reason:      Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "burn_id":                self.burn_id,
            "satellite_id":           self.satellite_id,
            "satellite_name":         self.satellite_name,
            "debris_id":              self.debris_id,
            "debris_designation":     self.debris_designation,
            "burn_time":              self.burn_time,          # ISO8601
            "burn_time_offset_s":     round(self.burn_time_offset_s, 1),
            "lead_time_to_tca_s":     round(self.lead_time_to_tca_s, 1),
            "delta_v_vector":         self.delta_v_km_s,       # ECI km/s
            "delta_v_m_s":            self.delta_v_m_s,        # ECI m/s
            "delta_v_mag_m_s":        round(self.delta_v_mag_m_s, 4),
            "burn_frame":             self.burn_frame,          # RTN m/s
            "maneuver_type":          self.maneuver_type,
            "fuel_cost_kg":           round(self.fuel_cost_kg, 4),
            "risk_level_before":      self.risk_level_before,
            "estimated_miss_after_km": round(self.estimated_miss_after_km, 3),
            "feasible":               self.feasible,
            "infeasible_reason":      self.infeasible_reason,
        }


# ═══════════════════════════════════════════════════════════════════════════
# RTN frame utilities
# ═══════════════════════════════════════════════════════════════════════════

def _rtn_basis(pos: np.ndarray, vel: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the Radial–Transverse–Normal unit vectors at a given ECI state.

    Returns
    -------
    R_hat : ndarray (3,)  – radial (outward from Earth centre)
    T_hat : ndarray (3,)  – transverse (prograde, along-track)
    N_hat : ndarray (3,)  – normal (orbit plane normal = h/|h|)
    """
    r_norm = np.linalg.norm(pos)
    if r_norm < 1e-6:
        raise ValueError("Position vector near zero; cannot compute RTN frame.")

    R_hat = pos / r_norm                                  # radial
    h     = np.cross(pos, vel)
    h_norm = np.linalg.norm(h)
    if h_norm < 1e-10:
        # Degenerate (rectilinear trajectory) – fall back to arbitrary normal
        N_hat = np.array([0.0, 0.0, 1.0])
    else:
        N_hat = h / h_norm                                # normal
    T_hat = np.cross(N_hat, R_hat)                        # transverse (prograde)
    return R_hat, T_hat, N_hat


def _eci_to_rtn(v_eci: np.ndarray, R: np.ndarray, T: np.ndarray, N: np.ndarray) -> np.ndarray:
    """Project an ECI vector into RTN components."""
    return np.array([np.dot(v_eci, R), np.dot(v_eci, T), np.dot(v_eci, N)])


def _rtn_to_eci(v_rtn: np.ndarray, R: np.ndarray, T: np.ndarray, N: np.ndarray) -> np.ndarray:
    """Convert an RTN vector back to ECI."""
    return v_rtn[0] * R + v_rtn[1] * T + v_rtn[2] * N


# ═══════════════════════════════════════════════════════════════════════════
# ΔV solver
# ═══════════════════════════════════════════════════════════════════════════

def _compute_avoidance_dv(
    sat_pos_now:    np.ndarray,   # ECI [km]
    sat_vel_now:    np.ndarray,   # ECI [km/s]
    sat_pos_tca:    np.ndarray,   # ECI [km]  – satellite position at TCA
    deb_pos_tca:    np.ndarray,   # ECI [km]  – debris   position at TCA
    sat_vel_tca:    np.ndarray,   # ECI [km/s]
    deb_vel_tca:    np.ndarray,   # ECI [km/s]
    tca_offset_s:   float,        # seconds until TCA
    lead_time_s:    float,        # burn fires lead_time_s before TCA
) -> Tuple[np.ndarray, str, float]:
    """
    Compute the optimal avoidance ΔV vector in ECI.

    Strategy
    --------
    1. Build RTN frame at current satellite position.
    2. Compute miss vector at TCA: m = deb_pos_tca − sat_pos_tca.
    3. Project m onto RTN to determine which axis dominates the threat geometry.
    4. Set avoidance direction OPPOSITE to the dominant miss component
       (push the satellite away from where the debris will be).
    5. Prefer T (transverse) over R (radial) for fuel efficiency, using T
       when |m_T| ≥ |m_R| (majority of LEO conjunctions).
    6. Compute required ΔV magnitude:
         ΔV ≈ SAFE_SEP_KM / (lead_time_s + ε)   [km/s]
       Reasoning: a tangential impulse δv applied lead_time_s before TCA
       shifts the satellite by approximately δv × lead_time_s in that
       direction by TCA.  We need the shift to be ≥ SAFE_SEP_KM − miss_now.
    7. Clamp to MAX_DV_KM_S.

    Returns
    -------
    dv_eci : ndarray (3,)  – ΔV vector in ECI [km/s]
    maneuver_type : str
    rtn_rtn_mag_ratio : float  – how much of ΔV is transverse vs radial
    """
    R, T, N = _rtn_basis(sat_pos_now, sat_vel_now)

    # Miss vector at TCA (from satellite to debris)
    miss_vec = deb_pos_tca - sat_pos_tca
    miss_km  = float(np.linalg.norm(miss_vec))

    # Project miss vector into RTN
    miss_rtn = _eci_to_rtn(miss_vec, R, T, N)
    m_R, m_T, m_N = miss_rtn

    # Required ΔV magnitude: shift by (SAFE_SEP_KM − miss_km) in lead_time_s
    gap_needed = max(0.0, SAFE_SEP_KM - miss_km)
    lt = max(30.0, lead_time_s)    # at least 30 s of lead time to avoid div/0
    dv_mag_needed = gap_needed / lt  # km/s

    # Cap at constraint
    dv_mag = min(dv_mag_needed, MAX_DV_KM_S)

    # Choose direction: prefer transverse, fall back to radial
    if abs(m_T) >= abs(m_R):
        # Transverse dominates — fire prograde or retrograde
        if m_T >= 0:
            direction_rtn = np.array([0.0, -1.0, 0.0])   # retrograde (move away from +T threat)
            maneuver_type = "RETROGRADE"
        else:
            direction_rtn = np.array([0.0, +1.0, 0.0])   # prograde
            maneuver_type = "PROGRADE"
    else:
        # Radial dominates — fire radial
        if m_R >= 0:
            direction_rtn = np.array([-1.0, 0.0, 0.0])   # radial minus
            maneuver_type = "RADIAL_MINUS"
        else:
            direction_rtn = np.array([+1.0, 0.0, 0.0])   # radial plus
            maneuver_type = "RADIAL_PLUS"

    dv_rtn = direction_rtn * dv_mag
    dv_eci = _rtn_to_eci(dv_rtn, R, T, N)

    return dv_eci, maneuver_type, dv_mag


def _estimated_miss_after(
    miss_km:     float,
    dv_mag_km_s: float,
    lead_time_s: float,
) -> float:
    """
    Rough linear estimate of miss distance after the avoidance burn.

    A tangential impulse dv applied lead_time_s before TCA shifts the
    satellite by approximately dv × lead_time_s [km].
    """
    shift = dv_mag_km_s * lead_time_s
    return math.sqrt(miss_km ** 2 + shift ** 2)


# ═══════════════════════════════════════════════════════════════════════════
# Avoidance Planner
# ═══════════════════════════════════════════════════════════════════════════

class AvoidancePlanner:
    """
    Autonomous collision avoidance planner.

    Consumes CDMs from ConjunctionDetectionService and generates burn plans
    satisfying:
      • ΔV ≤ 15 m/s per burn
      • ≥ 600 s cooldown between burns for the same satellite
      • Satellite has ≥ MIN_FUEL_KG remaining

    The planner maintains a per-satellite cooldown clock that persists across
    multiple ``plan()`` calls, so successive runs don't over-schedule burns.

    Parameters
    ----------
    risk_threshold : str
        Minimum CDM risk level to trigger planning.
        "CRITICAL" → only CRITICAL | "WARNING" → WARNING and above | etc.
    lead_time_s : float
        How many seconds before TCA to schedule the burn.
    """

    RISK_ORDER = {"NOMINAL": 0, "CAUTION": 1, "WARNING": 2, "CRITICAL": 3}

    def __init__(
        self,
        risk_threshold: str  = "WARNING",
        lead_time_s:    float = BURN_LEAD_TIME_S,
    ) -> None:
        self.risk_threshold = risk_threshold.upper()
        self.lead_time_s    = lead_time_s
        # cooldown_until[satellite_id] = datetime when next burn is allowed
        self._cooldown_until: Dict[str, datetime] = {}

    # ─── public entry point ─────────────────────────────────────────────────

    def plan(
        self,
        cdms:  List[ConjunctionDataMessage],
        epoch: Optional[datetime] = None,
    ) -> List[ManeuverPlan]:
        """
        Generate avoidance burn plans for all actionable CDMs.

        Parameters
        ----------
        cdms  : list[ConjunctionDataMessage]  – from ConjunctionDetectionService
        epoch : datetime | None               – current UTC time (defaults to now)

        Returns
        -------
        list[ManeuverPlan]
            All generated burn plans (both feasible and infeasible),
            sorted by risk level descending then miss_distance ascending.
        """
        epoch = epoch or datetime.utcnow()
        min_risk_order = self.RISK_ORDER.get(self.risk_threshold, 2)

        plans: List[ManeuverPlan] = []

        for cdm in cdms:
            # Filter by risk threshold
            risk_order = self.RISK_ORDER.get(cdm.risk_level, 0)
            if risk_order < min_risk_order:
                continue

            plan = self._plan_one(cdm, epoch)
            plans.append(plan)

        # Sort: CRITICAL first, then by miss distance
        plans.sort(key=lambda p: (
            -self.RISK_ORDER.get(p.risk_level_before, 0),
            p.estimated_miss_after_km,
        ))
        return plans

    def plan_from_registry_cdms(
        self,
        epoch: Optional[datetime] = None,
    ) -> List[ManeuverPlan]:
        """
        Pull CDMs from the shared conjunction service and plan burns.
        Requires a prior ``/cdm/run`` call.
        """
        from app.services.conjunction_service import get_service
        svc = get_service()
        cdms = svc.get_cdms(risk_level=None)
        return self.plan(cdms, epoch=epoch)

    def reset_cooldowns(self) -> None:
        """Clear all satellite cooldown records (for testing / simulation reset)."""
        self._cooldown_until.clear()

    def execute_plan(
        self,
        plan:         "ManeuverPlan",
        dry_mass_kg:  float = SAT_DRY_MASS_KG,
    ) -> dict:
        """
        Commit a feasible ManeuverPlan to the registry.

        Actions
        -------
        1. Validate the plan is feasible.
        2. Call :func:`apply_burn_to_registry` to deduct fuel atomically.
        3. Update satellite velocity in the registry by adding ΔV.
        4. Return a confirmation dict with the BurnRecord.

        Parameters
        ----------
        plan        : ManeuverPlan – must have feasible=True
        dry_mass_kg : float        – satellite dry mass estimate [kg]

        Raises
        ------
        ValueError            if plan.feasible is False
        InsufficientFuelError if fuel check fails at execution time
        KeyError              if satellite not in registry
        """
        if not plan.feasible:
            raise ValueError(
                f"Cannot execute infeasible plan: {plan.infeasible_reason}"
            )

        from app.data.registry import get_satellite, fast_update_satellite

        sat = get_satellite(plan.satellite_id)
        if sat is None:
            raise KeyError(f"Satellite '{plan.satellite_id}' not found")

        # Commit fuel deduction (also re-validates fuel at execution time)
        record = apply_burn_to_registry(
            satellite_id = plan.satellite_id,
            burn_id      = plan.burn_id,
            delta_v_m_s  = plan.delta_v_mag_m_s,
            dry_mass_kg  = dry_mass_kg,
            isp_s        = ISP_S,
        )

        # Apply ΔV to satellite velocity (must supply all 6 pos+vel args)
        dv  = plan.delta_v_km_s
        pos = sat.position
        vel = sat.velocity
        fast_update_satellite(
            plan.satellite_id,
            pos.x, pos.y, pos.z,
            vel.vx + dv["x"],
            vel.vy + dv["y"],
            vel.vz + dv["z"],
        )

        logger.info(
            "ManeuverPlan executed: sat=%s  burn=%s  ΔV=%.3f m/s  "
            "fuel: %.3f → %.3f kg",
            plan.satellite_id, plan.burn_id,
            plan.delta_v_mag_m_s,
            record.fuel_before, record.fuel_after,
        )
        return {
            "status":      "executed",
            "burn_record": record.to_dict(),
            "fuel_before_kg": record.fuel_before,
            "fuel_after_kg":  record.fuel_after,
            "propellant_used_kg": record.propellant_used,
        }

    # ─── per-CDM planning ────────────────────────────────────────────────────

    def _plan_one(
        self,
        cdm:   ConjunctionDataMessage,
        epoch: datetime,
    ) -> ManeuverPlan:
        """Plan one avoidance burn for a single CDM."""
        from app.data.registry import get_satellite

        burn_id = str(uuid.uuid4())

        # ── Fetch satellite from registry ────────────────────────────────────
        sat = get_satellite(cdm.satellite_id)
        if sat is None:
            return self._infeasible(
                burn_id, cdm, epoch,
                f"Satellite '{cdm.satellite_id}' not found in registry.",
            )

        # ── Check minimum fuel ───────────────────────────────────────────────
        if sat.fuel_kg < MIN_FUEL_KG:
            return self._infeasible(
                burn_id, cdm, epoch,
                f"Insufficient fuel: {sat.fuel_kg:.2f} kg < {MIN_FUEL_KG} kg minimum.",
            )

        # ── Check cooldown ───────────────────────────────────────────────────
        cooldown_expires = self._cooldown_until.get(cdm.satellite_id)
        if cooldown_expires and epoch < cooldown_expires:
            remaining = (cooldown_expires - epoch).total_seconds()
            return self._infeasible(
                burn_id, cdm, epoch,
                f"Cooldown active: {remaining:.0f} s remaining before next burn.",
            )

        # ── Read satellite current state ─────────────────────────────────────
        sat_pos_now = np.array([sat.position.x, sat.position.y, sat.position.z])
        sat_vel_now = np.array([sat.velocity.vx, sat.velocity.vy, sat.velocity.vz])

        # ── Extract TCA positions from CDM ───────────────────────────────────
        sat_pos_tca = np.array(cdm.sat_pos_at_tca, dtype=np.float64)
        deb_pos_tca = np.array(cdm.deb_pos_at_tca, dtype=np.float64)

        # Approximate velocities at TCA via linear propagation
        tca_offset = cdm.tca_offset_s
        sat_vel_tca = sat_vel_now   # first-order: velocity doesn't change much in LEO over ~min
        deb_vel_tca = sat_vel_now - np.array([0.0, cdm.relative_speed_km_s, 0.0])

        # ── Compute avoidance ΔV ─────────────────────────────────────────────
        lead_t = min(self.lead_time_s, max(30.0, tca_offset - 60.0))

        try:
            dv_eci, maneuver_type, dv_mag = _compute_avoidance_dv(
                sat_pos_now, sat_vel_now,
                sat_pos_tca, deb_pos_tca,
                sat_vel_tca, deb_vel_tca,
                tca_offset, lead_t,
            )
        except Exception as exc:
            return self._infeasible(burn_id, cdm, epoch, f"ΔV computation failed: {exc}")

        dv_mag_m_s = float(np.linalg.norm(dv_eci)) * 1000.0  # km/s → m/s

        # ── Fuel feasibility check (Tsiolkovsky, Isp=300 s) ──────────────────
        #    wet mass = current fuel + dry-mass estimate
        wet_mass  = sat.fuel_kg + SAT_DRY_MASS_KG
        ok, fuel_cost, reason = _fuel_is_feasible(
            sat.fuel_kg, wet_mass, dv_mag_m_s, isp_s=ISP_S,
        )
        if not ok:
            return self._infeasible(burn_id, cdm, epoch, reason)

        # ── RTN decomposition for readout ────────────────────────────────────
        R, T, N = _rtn_basis(sat_pos_now, sat_vel_now)
        dv_rtn  = _eci_to_rtn(dv_eci, R, T, N)
        burn_frame = {
            "R_m_s": round(float(dv_rtn[0]) * 1000.0, 4),
            "T_m_s": round(float(dv_rtn[1]) * 1000.0, 4),
            "N_m_s": round(float(dv_rtn[2]) * 1000.0, 4),
        }

        # ── Burn time ────────────────────────────────────────────────────────
        burn_offset_s = max(0.0, tca_offset - lead_t)
        burn_dt = epoch + timedelta(seconds=burn_offset_s)

        # ── Estimated miss after burn ─────────────────────────────────────────
        est_miss = _estimated_miss_after(cdm.miss_distance_km, float(np.linalg.norm(dv_eci)), lead_t)

        # ── Register cooldown ────────────────────────────────────────────────
        self._cooldown_until[cdm.satellite_id] = burn_dt + timedelta(seconds=COOLDOWN_S)

        return ManeuverPlan(
            burn_id              = burn_id,
            satellite_id         = cdm.satellite_id,
            satellite_name       = cdm.satellite_name,
            debris_id            = cdm.debris_id,
            debris_designation   = cdm.debris_designation,
            burn_time            = burn_dt.isoformat(),
            burn_time_offset_s   = burn_offset_s,
            lead_time_to_tca_s   = lead_t,
            delta_v_km_s         = {"x": round(float(dv_eci[0]), 8),
                                     "y": round(float(dv_eci[1]), 8),
                                     "z": round(float(dv_eci[2]), 8)},
            delta_v_m_s          = {"x": round(float(dv_eci[0]) * 1000, 4),
                                     "y": round(float(dv_eci[1]) * 1000, 4),
                                     "z": round(float(dv_eci[2]) * 1000, 4)},
            delta_v_mag_m_s      = round(dv_mag_m_s, 4),
            burn_frame           = burn_frame,
            maneuver_type        = maneuver_type,
            fuel_cost_kg         = round(fuel_cost, 4),
            risk_level_before    = cdm.risk_level,
            estimated_miss_after_km = round(est_miss, 3),
            feasible             = True,
        )

    # ─── helper ─────────────────────────────────────────────────────────────

    def _infeasible(
        self,
        burn_id: str,
        cdm:     ConjunctionDataMessage,
        epoch:   datetime,
        reason:  str,
    ) -> ManeuverPlan:
        logger.warning("Avoidance plan infeasible  sat=%s  deb=%s  reason=%s",
                       cdm.satellite_id, cdm.debris_id, reason)
        return ManeuverPlan(
            burn_id              = burn_id,
            satellite_id         = cdm.satellite_id,
            satellite_name       = cdm.satellite_name,
            debris_id            = cdm.debris_id,
            debris_designation   = cdm.debris_designation,
            burn_time            = epoch.isoformat(),
            burn_time_offset_s   = 0.0,
            lead_time_to_tca_s   = 0.0,
            delta_v_km_s         = {"x": 0.0, "y": 0.0, "z": 0.0},
            delta_v_m_s          = {"x": 0.0, "y": 0.0, "z": 0.0},
            delta_v_mag_m_s      = 0.0,
            burn_frame           = {"R_m_s": 0.0, "T_m_s": 0.0, "N_m_s": 0.0},
            maneuver_type        = "NONE",
            fuel_cost_kg         = 0.0,
            risk_level_before    = cdm.risk_level,
            estimated_miss_after_km = cdm.miss_distance_km,
            feasible             = False,
            infeasible_reason    = reason,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Module-level singleton
# ═══════════════════════════════════════════════════════════════════════════

_planner: Optional[AvoidancePlanner] = None


def get_planner() -> AvoidancePlanner:
    global _planner
    if _planner is None:
        _planner = AvoidancePlanner()
    return _planner


def reset_planner(
    risk_threshold: str   = "WARNING",
    lead_time_s:    float = BURN_LEAD_TIME_S,
) -> AvoidancePlanner:
    global _planner
    _planner = AvoidancePlanner(risk_threshold=risk_threshold, lead_time_s=lead_time_s)
    return _planner
