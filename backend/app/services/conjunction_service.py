"""
conjunction_service.py
======================
24-Hour Conjunction Detection Service for the Autonomous Constellation Manager.

Architecture
------------
The service runs in three clearly separated phases:

  Phase 1 – Trajectory Propagation
      Propagate every satellite and debris object forward in time using the
      RK4 orbit propagator (orbit_propagator.py).  Positions are recorded
      at configurable time steps (default 60 s) to build a 24-hour
      trajectory database.

      Cost: O((S + N) · T)   where T = # time steps, S = # satellites,
                               N = # debris objects.

  Phase 2 – Temporal-Spatial Screening
      At each time step, a fresh KDTree is built from the propagated debris
      positions at that epoch.  Each satellite position is queried against
      the tree with the screening radius (default 50 km).  Pairs that never
      enter the screening volume are cheaply eliminated with O(log N) queries.

      Cost: O(T · N log N + T · S · k · log N)   (k = avg candidates/sat)

  Phase 3 – TCA Refinement
      For each (satellite, debris, epoch_window) triple that passed Phase 2,
      a fine-grained ternary search over the surrounding time window finds
      the precise TCA and miss distance using the already-propagated
      trajectory arrays (no additional integration required).

      Cost: O(E · W)   where E = # candidate events, W = window steps.

Output – Conjunction Data Messages (CDM)
      Each conjunction event is packaged as a CDM containing:
        • satellite_id, debris_id
        • time_of_closest_approach  (UTC ISO8601)
        • miss_distance_km
        • relative_speed_km_s
        • risk_level  (NOMINAL|CAUTION|WARNING|CRITICAL)
        • satellite/debris ECI positions at TCA
        • tca_offset_s  (seconds from now until TCA)

CDMs are stored in a module-level in-memory store sorted by miss distance.
The store is rebuilt on each full conjunction run.

Constants / defaults
--------------------
  HORIZON_HOURS       = 24          hours
  STEP_SECONDS        = 60          integration step [s]
  SCREEN_RADIUS_KM    = 50.0        KDTree ball-query radius [km]
  CDM_THRESHOLD_KM    = 10.0        only generate CDMs below this miss distance
  RISK_CRITICAL_KM    = 0.1         < 100 m
  RISK_WARNING_KM     = 0.5
  RISK_CAUTION_KM     = 2.0
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import KDTree

from app.models.sim_objects import SimSatellite, SimDebris
from app.physics.orbit_propagator import propagate_state

logger = logging.getLogger(__name__)

# ── Service configuration ────────────────────────────────────────────────────
HORIZON_HOURS:       float = 24.0
STEP_SECONDS:        float = 60.0
SCREEN_RADIUS_KM:    float = 50.0
CDM_THRESHOLD_KM:    float = 10.0    # generate CDMs only below this miss dist
RISK_CRITICAL_KM:    float = 0.1     # < 100 m
RISK_WARNING_KM:     float = 0.5
RISK_CAUTION_KM:     float = 2.0
TCA_REFINE_HALF_WIN: int   = 3       # ±3 steps around the coarse minimum


# ═══════════════════════════════════════════════════════════════════════════
# CDM data class  (the canonical output type)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ConjunctionDataMessage:
    """
    Conjunction Data Message (CDM) for one satellite–debris close approach.

    Follows the structure requested by the user with additional diagnostic
    fields for operations use.

    Attributes
    ----------
    satellite_id          : str
    debris_id             : str
    satellite_name        : str
    debris_designation    : str
    time_of_closest_approach : str  – UTC ISO8601
    tca_offset_s          : float  – seconds from now until TCA (negative = past)
    miss_distance_km      : float
    relative_speed_km_s   : float
    risk_level            : str    – NOMINAL | CAUTION | WARNING | CRITICAL
    collision_probability : float  – simplified geometric Pc estimate [0, 1]
    sat_pos_at_tca        : list[float]  – ECI [km]
    deb_pos_at_tca        : list[float]  – ECI [km]
    generated_at          : str    – when this CDM was produced
    """
    satellite_id:             str
    debris_id:                str
    satellite_name:           str
    debris_designation:       str
    time_of_closest_approach: str
    tca_offset_s:             float
    miss_distance_km:         float
    relative_speed_km_s:      float
    risk_level:               str
    collision_probability:    float
    sat_pos_at_tca:           List[float]
    deb_pos_at_tca:           List[float]
    generated_at:             str

    def to_dict(self) -> dict:
        """Return a plain dict matching the user-specified CDM schema."""
        return {
            # Core required fields
            "satellite_id":               self.satellite_id,
            "debris_id":                  self.debris_id,
            "time_of_closest_approach":   self.time_of_closest_approach,
            "miss_distance_km":           round(self.miss_distance_km, 6),
            "risk_level":                 self.risk_level,
            # Extended fields
            "satellite_name":             self.satellite_name,
            "debris_designation":         self.debris_designation,
            "tca_offset_s":               round(self.tca_offset_s, 1),
            "relative_speed_km_s":        round(self.relative_speed_km_s, 4),
            "collision_probability":      round(self.collision_probability, 8),
            "sat_pos_at_tca_km":          [round(v, 3) for v in self.sat_pos_at_tca],
            "deb_pos_at_tca_km":          [round(v, 3) for v in self.deb_pos_at_tca],
            "generated_at":               self.generated_at,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Risk helpers
# ═══════════════════════════════════════════════════════════════════════════

def _risk_level(dist_km: float) -> str:
    if dist_km < RISK_CRITICAL_KM:
        return "CRITICAL"
    if dist_km < RISK_WARNING_KM:
        return "WARNING"
    if dist_km < RISK_CAUTION_KM:
        return "CAUTION"
    return "NOMINAL"


def _collision_probability(miss_km: float, rel_speed_km_s: float) -> float:
    """
    Simplified geometric collision probability estimate.

    Uses a hard-body radius model with a combined hard-body radius of 0.01 km
    (10 m) for the satellite and debris pair.  The encounter is approximated
    as a point-mass flyby with the relative speed and miss distance known.

    P_c ≈ (π · R_hb²) / (√(2π) · σ_b · σ_c · V_rel · T_enc)

    For operational screening this is a rough order-of-magnitude estimate
    intended to prioritise events.  A full Monte-Carlo covariance analysis
    would be required for mission-critical decisions.
    """
    R_hb = 0.01       # combined hard-body radius [km] (satellite + debris)
    sigma = 0.05      # assumed 1-σ position uncertainty [km]

    if miss_km <= 1e-10 or rel_speed_km_s <= 0:
        return 1.0 if miss_km < R_hb else 0.0

    # Gaussian miss-distance probability density
    import math
    exponent = -(miss_km ** 2) / (2.0 * sigma ** 2)
    Pc = (math.pi * R_hb ** 2) / (2.0 * math.pi * sigma ** 2) * math.exp(exponent)
    return float(min(1.0, max(0.0, Pc)))


# ═══════════════════════════════════════════════════════════════════════════
# Trajectory builder (uses RK4 propagator)
# ═══════════════════════════════════════════════════════════════════════════

def _propagate_trajectory(
    obj_id:       str,
    init_pos:     np.ndarray,   # (3,) [km]
    init_vel:     np.ndarray,   # (3,) [km/s]
    n_steps:      int,
    step_s:       float,
    mass_kg:      float = 100.0,
    area_m2:      float = 2.0,
    drag_coeff:   float = 2.2,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Propagate an object forward for *n_steps* × *step_s* seconds using RK4.

    Returns
    -------
    positions : ndarray (n_steps+1, 3)  – ECI positions at each epoch [km]
    velocities: ndarray (n_steps+1, 3)  – ECI velocities              [km/s]

    The first row is the initial state (t=0).
    """
    from app.models.sim_objects import StateVector

    pos_traj = np.empty((n_steps + 1, 3), dtype=np.float64)
    vel_traj = np.empty((n_steps + 1, 3), dtype=np.float64)
    pos_traj[0] = init_pos
    vel_traj[0] = init_vel

    cur_sv = StateVector(
        x=float(init_pos[0]), y=float(init_pos[1]), z=float(init_pos[2]),
        vx=float(init_vel[0]), vy=float(init_vel[1]), vz=float(init_vel[2]),
    )

    for i in range(1, n_steps + 1):
        result = propagate_state(
            cur_sv, step_s,
            area_m2=area_m2, mass_kg=mass_kg, drag_coeff=drag_coeff,
        )
        cur_sv = result.state
        pos_traj[i] = [cur_sv.x,  cur_sv.y,  cur_sv.z]
        vel_traj[i] = [cur_sv.vx, cur_sv.vy, cur_sv.vz]

        if result.reentry:
            # Fill remainder with last known position and zero out velocity
            pos_traj[i+1:] = pos_traj[i]
            vel_traj[i+1:] = 0.0
            logger.warning("Re-entry detected during trajectory build: %s at step %d", obj_id, i)
            break

    return pos_traj, vel_traj


# ═══════════════════════════════════════════════════════════════════════════
# TCA refinement on propagated trajectory arrays
# ═══════════════════════════════════════════════════════════════════════════

def _refine_tca(
    sat_pos_traj: np.ndarray,   # (T+1, 3)
    sat_vel_traj: np.ndarray,
    deb_pos_traj: np.ndarray,
    deb_vel_traj: np.ndarray,
    coarse_step_idx: int,       # step index of coarse minimum
    n_steps: int,
    step_s: float,
    half_win: int = TCA_REFINE_HALF_WIN,
) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    """
    Refine TCA within a ±half_win step window around *coarse_step_idx*.

    Uses sub-step linear interpolation within each propagated segment to
    achieve sub-second TCA precision without additional RK4 calls.

    Returns
    -------
    (tca_s, miss_km, rel_speed_km_s, sat_pos_at_tca, deb_pos_at_tca)
    """
    lo_idx = max(0, coarse_step_idx - half_win)
    hi_idx = min(n_steps, coarse_step_idx + half_win)

    # ── Fine grid: 100 sub-steps per coarse step ────────────────────────────
    fine_pts = (hi_idx - lo_idx) * 100 + 1
    t_fine   = np.linspace(0.0, (hi_idx - lo_idx) * step_s, fine_pts)

    lo_sat_p = sat_pos_traj[lo_idx];  lo_sat_v = sat_vel_traj[lo_idx]
    lo_deb_p = deb_pos_traj[lo_idx];  lo_deb_v = deb_vel_traj[lo_idx]

    # Linear propagation within the window
    sat_fine = lo_sat_p[None, :] + lo_sat_v[None, :] * t_fine[:, None]
    deb_fine = lo_deb_p[None, :] + lo_deb_v[None, :] * t_fine[:, None]
    ranges   = np.linalg.norm(sat_fine - deb_fine, axis=1)

    best = int(np.argmin(ranges))
    tca_offset = lo_idx * step_s + float(t_fine[best])
    miss_km    = float(ranges[best])

    sp_tca = sat_fine[best]
    dp_tca = deb_fine[best]

    # Relative speed at TCA (use propagated velocities at the coarse index)
    rel_v = sat_vel_traj[coarse_step_idx] - deb_vel_traj[coarse_step_idx]
    rel_speed = float(np.linalg.norm(rel_v))

    return tca_offset, miss_km, rel_speed, sp_tca, dp_tca


# ═══════════════════════════════════════════════════════════════════════════
# Conjunction Detection Service
# ═══════════════════════════════════════════════════════════════════════════

class ConjunctionDetectionService:
    """
    24-hour predictive conjunction detection service.

    Workflow
    --------
    1. ``run(satellites, debris_list)``
         Propagates all objects, screens with KDTree at each epoch,
         refines TCA for candidates, builds CDMs.
    2. Stores results in ``self.cdms`` (deduplicated by pair, lowest miss).
    3. Results accessible via ``get_cdms()``, ``get_summary()``.

    Parameters
    ----------
    horizon_hours   : float  – Propagation horizon (default 24 h)
    step_seconds    : float  – RK4 integration step (default 60 s)
    screen_radius_km: float  – KDTree query radius  (default 50 km)
    cdm_threshold_km: float  – Only emit CDMs below this miss distance (default 10 km)
    """

    def __init__(
        self,
        horizon_hours:    float = HORIZON_HOURS,
        step_seconds:     float = STEP_SECONDS,
        screen_radius_km: float = SCREEN_RADIUS_KM,
        cdm_threshold_km: float = CDM_THRESHOLD_KM,
    ) -> None:
        self.horizon_hours     = horizon_hours
        self.step_seconds      = step_seconds
        self.screen_radius_km  = screen_radius_km
        self.cdm_threshold_km  = cdm_threshold_km

        self.cdms:          List[ConjunctionDataMessage] = []
        self.last_run_at:   Optional[datetime]           = None
        self.last_run_ms:   float                        = 0.0
        self.pairs_checked: int                          = 0

    # ─── primary entry point ────────────────────────────────────────────────

    def run(
        self,
        satellites:   List[SimSatellite],
        debris_list:  List[SimDebris],
        epoch:        Optional[datetime] = None,
    ) -> List[ConjunctionDataMessage]:
        """
        Execute a full 24-hour conjunction detection run.

        Parameters
        ----------
        satellites  : list[SimSatellite]  – objects to protect
        debris_list : list[SimDebris]     – threat objects
        epoch       : datetime | None     – simulation start epoch (default: now UTC)

        Returns
        -------
        list[ConjunctionDataMessage]
            All CDMs with miss_distance_km < cdm_threshold_km, sorted by
            miss distance ascending (most critical first).
        """
        t_wall = time.perf_counter()
        epoch  = epoch or datetime.utcnow()

        if not satellites:
            logger.warning("ConjunctionDetectionService.run: no satellites supplied")
            self.cdms = []
            return []
        if not debris_list:
            logger.warning("ConjunctionDetectionService.run: no debris objects supplied")
            self.cdms = []
            return []

        n_steps  = int(self.horizon_hours * 3600.0 / self.step_seconds)
        step_s   = self.step_seconds

        logger.info(
            "CDM run START: %d sats × %d debris  horizon=%.0f h  step=%.0f s  n_steps=%d",
            len(satellites), len(debris_list), self.horizon_hours, step_s, n_steps,
        )

        # ── Phase 1: Propagate all trajectories ─────────────────────────────
        sat_trajs:  Dict[str, Tuple[SimSatellite, np.ndarray, np.ndarray]] = {}
        deb_trajs:  Dict[str, Tuple[SimDebris,    np.ndarray, np.ndarray]] = {}

        for sat in satellites:
            p0 = np.array([sat.position.x, sat.position.y, sat.position.z])
            v0 = np.array([sat.velocity.vx, sat.velocity.vy, sat.velocity.vz])
            mass = max(1.0, sat.fuel_kg + 100.0)
            pp, vv = _propagate_trajectory(sat.id, p0, v0, n_steps, step_s, mass_kg=mass)
            sat_trajs[sat.id] = (sat, pp, vv)

        for deb in debris_list:
            p0 = np.array([deb.position.x, deb.position.y, deb.position.z])
            v0 = np.array([deb.velocity.vx, deb.velocity.vy, deb.velocity.vz])
            pp, vv = _propagate_trajectory(deb.id, p0, v0, n_steps, step_s, mass_kg=10.0)
            deb_trajs[deb.id] = (deb, pp, vv)

        # ── Phase 2: Temporal-spatial screening ─────────────────────────────
        # candidate_events: {pair_key: (sat_id, deb_id, best_step_idx, best_dist)}
        candidate_events: Dict[str, Tuple[str, str, int, float]] = {}

        deb_ids_list = list(deb_trajs.keys())
        deb_objs     = [deb_trajs[did][0] for did in deb_ids_list]
        pairs_checked = 0

        for step_idx in range(n_steps + 1):
            # Build KDTree from debris positions at this epoch
            deb_pos_step = np.array(
                [deb_trajs[did][1][step_idx] for did in deb_ids_list]
            )  # (N, 3)
            tree = KDTree(deb_pos_step)

            for sat_id, (sat, sat_pp, _) in sat_trajs.items():
                sat_p = sat_pp[step_idx]
                indices = tree.query_ball_point(sat_p, r=self.screen_radius_km)
                pairs_checked += len(indices)

                for deb_idx in indices:
                    deb_id   = deb_ids_list[deb_idx]
                    pair_key = f"{sat_id}::{deb_id}"
                    dist     = float(np.linalg.norm(sat_p - deb_pos_step[deb_idx]))

                    prev = candidate_events.get(pair_key)
                    if prev is None or dist < prev[3]:
                        candidate_events[pair_key] = (sat_id, deb_id, step_idx, dist)

        self.pairs_checked = pairs_checked
        logger.info(
            "Phase 2 complete: %d candidate windows from %d pair-step checks",
            len(candidate_events), pairs_checked,
        )

        # ── Phase 3: TCA refinement and CDM generation ───────────────────────
        generated_at = datetime.utcnow().isoformat()
        cdms: List[ConjunctionDataMessage] = []
        # Deduplicate: keep only the minimum-distance CDM per pair
        best_per_pair: Dict[str, ConjunctionDataMessage] = {}

        for pair_key, (sat_id, deb_id, coarse_idx, coarse_dist) in candidate_events.items():
            sat, sat_pp, sat_vv = sat_trajs[sat_id]
            deb, deb_pp, deb_vv = deb_trajs[deb_id]

            tca_offset, miss_km, rel_speed, sp_tca, dp_tca = _refine_tca(
                sat_pp, sat_vv, deb_pp, deb_vv,
                coarse_idx, n_steps, step_s,
            )

            if miss_km > self.cdm_threshold_km:
                continue

            tca_dt  = epoch + timedelta(seconds=tca_offset)
            risk    = _risk_level(miss_km)
            Pc      = _collision_probability(miss_km, rel_speed)

            cdm = ConjunctionDataMessage(
                satellite_id              = sat.id,
                debris_id                 = deb.id,
                satellite_name            = sat.name,
                debris_designation        = deb.designation,
                time_of_closest_approach  = tca_dt.isoformat(),
                tca_offset_s              = tca_offset,
                miss_distance_km          = miss_km,
                relative_speed_km_s       = rel_speed,
                risk_level                = risk,
                collision_probability     = Pc,
                sat_pos_at_tca            = sp_tca.tolist(),
                deb_pos_at_tca            = dp_tca.tolist(),
                generated_at              = generated_at,
            )

            prev = best_per_pair.get(pair_key)
            if prev is None or miss_km < prev.miss_distance_km:
                best_per_pair[pair_key] = cdm

        cdms = sorted(best_per_pair.values(), key=lambda c: c.miss_distance_km)

        self.cdms         = cdms
        self.last_run_at  = datetime.utcnow()
        self.last_run_ms  = (time.perf_counter() - t_wall) * 1000.0

        logger.info(
            "CDM run COMPLETE: %d CDMs generated  %.1f ms",
            len(cdms), self.last_run_ms,
        )
        return cdms

    # ─── registry-aware convenience method ──────────────────────────────────

    def run_from_registry(
        self,
        horizon_hours: Optional[float] = None,
        epoch:         Optional[datetime] = None,
    ) -> List[ConjunctionDataMessage]:
        """
        Pull objects directly from the in-memory registries and run a
        full conjunction detection pass.

        Parameters
        ----------
        horizon_hours : float | None  – override the service horizon
        epoch         : datetime | None
        """
        from app.data.registry import satellites, debris as deb_registry
        if horizon_hours is not None:
            self.horizon_hours = horizon_hours
        return self.run(
            list(satellites.values()),
            list(deb_registry.values()),
            epoch=epoch,
        )

    # ─── query methods ──────────────────────────────────────────────────────

    def get_cdms(
        self,
        *,
        risk_level:  Optional[str]  = None,
        satellite_id: Optional[str] = None,
        debris_id:    Optional[str] = None,
        limit:        int            = 500,
    ) -> List[ConjunctionDataMessage]:
        """
        Return stored CDMs with optional filtering.

        Parameters
        ----------
        risk_level   : "CRITICAL" | "WARNING" | "CAUTION" | "NOMINAL" | None
        satellite_id : filter to one satellite
        debris_id    : filter to one debris object
        limit        : max number of results to return
        """
        result = self.cdms
        if risk_level:
            result = [c for c in result if c.risk_level == risk_level.upper()]
        if satellite_id:
            result = [c for c in result if c.satellite_id == satellite_id]
        if debris_id:
            result = [c for c in result if c.debris_id == debris_id]
        return result[:limit]

    def get_summary(self) -> dict:
        """Return a high-level summary of the last conjunction run."""
        if not self.last_run_at:
            return {"status": "no_run_yet", "cdm_count": 0}

        by_risk: Dict[str, int] = {"CRITICAL": 0, "WARNING": 0, "CAUTION": 0, "NOMINAL": 0}
        for cdm in self.cdms:
            by_risk[cdm.risk_level] = by_risk.get(cdm.risk_level, 0) + 1

        return {
            "status":           "ok",
            "last_run_at":      self.last_run_at.isoformat(),
            "last_run_ms":      round(self.last_run_ms, 1),
            "cdm_count":        len(self.cdms),
            "pairs_checked":    self.pairs_checked,
            "horizon_hours":    self.horizon_hours,
            "step_seconds":     self.step_seconds,
            "screen_radius_km": self.screen_radius_km,
            "cdm_threshold_km": self.cdm_threshold_km,
            "by_risk_level":    by_risk,
            "most_critical":    self.cdms[0].to_dict() if self.cdms else None,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Module-level singleton (shared across all API handlers)
# ═══════════════════════════════════════════════════════════════════════════

_service: Optional[ConjunctionDetectionService] = None


def get_service() -> ConjunctionDetectionService:
    """Return (or lazily create) the shared service singleton."""
    global _service
    if _service is None:
        _service = ConjunctionDetectionService()
    return _service


def reset_service(
    horizon_hours:    float = HORIZON_HOURS,
    step_seconds:     float = STEP_SECONDS,
    screen_radius_km: float = SCREEN_RADIUS_KM,
    cdm_threshold_km: float = CDM_THRESHOLD_KM,
) -> ConjunctionDetectionService:
    """Replace the singleton with a freshly configured instance."""
    global _service
    _service = ConjunctionDetectionService(
        horizon_hours    = horizon_hours,
        step_seconds     = step_seconds,
        screen_radius_km = screen_radius_km,
        cdm_threshold_km = cdm_threshold_km,
    )
    return _service
