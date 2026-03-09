"""
spatial_index.py
================
Spatial indexing and collision detection system for the Autonomous
Constellation Manager.

Overview
--------
Satellites and debris travel at ~7–8 km/s in LEO.  A naive O(N·M) distance
check between every satellite–debris pair is too slow for large constellations
at high update rates.  This module uses a **KDTree** (scipy.spatial.KDTree)
built from debris ECI positions [km] to reduce each satellite's nearest-
neighbour query to O(k · log N) — where k is the number of candidates
returned within the search radius.

Two-phase architecture
----------------------
Phase 1 – Spatial pre-filter  (fast, cheap)
    build_spatial_index(debris_positions) → SpatialIndex
    find_nearby_objects(sat_position)     → List[NearbyObject]

    A KDTree ball-query with *query_radius_km* (default 50 km) returns only
    the debris candidates worth examining.  The vast majority of pairs are
    eliminated in O(log N) with a single tree query.

Phase 2 – TCA refinement  (only for candidates)
    compute_tca(sat_state, debris_state, horizon_s) → TCAResult

    For each candidate pair, a short-arc linear propagation finds the
    **Time of Closest Approach** (TCA) by minimising the inter-object
    range.  A golden-section / ternary-search over the look-ahead horizon
    gives sub-second TCA precision at minimal CPU cost.

Collision risk flag
-------------------
A pair is flagged as a **collision risk** when the miss distance at TCA
is below *COLLISION_THRESHOLD_KM* (default 0.1 km = 100 m).

Public API summary
------------------
    build_spatial_index(debris_positions, *, debris_ids, query_radius_km)
        → SpatialIndex

    find_nearby_objects(satellite_position, *, satellite_velocity)
        → List[NearbyObject]          (called on a SpatialIndex instance)

    compute_tca(sat_pos, sat_vel, deb_pos, deb_vel, *, horizon_s, steps)
        → TCAResult

    screen_satellite_vs_debris(satellite, debris_list, *, radius_km, horizon_s)
        → CollisionScreenResult       (convenience full pipeline)

    screen_all(*, query_radius_km, tca_horizon_s)
        → List[CollisionScreenResult] (registry-aware full constellation screen)

Units
-----
    Positions  [km]   ECI J2000
    Velocities [km/s] ECI J2000
    Time       [s]
    Mass       [kg]
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.spatial import KDTree

from app.models.sim_objects import SimSatellite, SimDebris, StateVector

logger = logging.getLogger(__name__)

# ── Configurable thresholds ──────────────────────────────────────────────────
DEFAULT_QUERY_RADIUS_KM:   float = 50.0    # Phase-1 spatial pre-filter radius
COLLISION_THRESHOLD_KM:    float = 0.1     # 100 m – flag as collision risk
DEFAULT_TCA_HORIZON_S:     float = 300.0   # 5-minute linear look-ahead
DEFAULT_TCA_STEPS:         int   = 500     # grid resolution for TCA search
TERNARY_ITERATIONS:        int   = 40      # golden-section refinement iterations


# ═══════════════════════════════════════════════════════════════════════════
# Result data-classes
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class NearbyObject:
    """
    A debris object returned by the Phase-1 spatial query.

    Attributes
    ----------
    debris_id       : str   – registry ID of the debris object
    debris_index    : int   – index into the original debris list
    current_dist_km : float – instantaneous Euclidean distance  [km]
    position        : np.ndarray shape (3,) – ECI position  [km]
    velocity        : np.ndarray shape (3,) – ECI velocity  [km/s]
    """
    debris_id:        str
    debris_index:     int
    current_dist_km:  float
    position:         np.ndarray
    velocity:         np.ndarray


@dataclass
class TCAResult:
    """
    Time of Closest Approach between one satellite and one debris object.

    Attributes
    ----------
    tca_s           : float  – time to closest approach from *now*  [s]
    miss_distance_km: float  – range at TCA  [km]
    collision_risk  : bool   – True when miss_distance_km < COLLISION_THRESHOLD_KM
    sat_pos_at_tca  : np.ndarray (3,)
    deb_pos_at_tca  : np.ndarray (3,)
    risk_level      : str    – "NOMINAL" | "CAUTION" | "WARNING" | "CRITICAL"
    """
    tca_s:           float
    miss_distance_km: float
    collision_risk:  bool
    sat_pos_at_tca:  np.ndarray
    deb_pos_at_tca:  np.ndarray
    risk_level:      str

    def to_dict(self) -> dict:
        return {
            "tca_s":            round(self.tca_s, 3),
            "miss_distance_km": round(self.miss_distance_km, 6),
            "collision_risk":   self.collision_risk,
            "risk_level":       self.risk_level,
            "sat_pos_at_tca":   self.sat_pos_at_tca.tolist(),
            "deb_pos_at_tca":   self.deb_pos_at_tca.tolist(),
        }


@dataclass
class CollisionScreenResult:
    """
    Full collision screening result for one satellite.

    Attributes
    ----------
    satellite_id    : str
    satellite_name  : str
    query_radius_km : float
    candidates_found: int          – objects within query radius
    risks           : list[dict]   – one entry per collision-risk pair
    warnings        : list[dict]   – pairs with miss < 2 km (non-critical)
    screen_time_ms  : float        – wall-clock time for this satellite
    """
    satellite_id:    str
    satellite_name:  str
    query_radius_km: float
    candidates_found: int
    risks:           List[dict]
    warnings:        List[dict]
    screen_time_ms:  float

    @property
    def has_collision_risk(self) -> bool:
        return len(self.risks) > 0

    def to_dict(self) -> dict:
        return {
            "satellite_id":     self.satellite_id,
            "satellite_name":   self.satellite_name,
            "query_radius_km":  self.query_radius_km,
            "candidates_found": self.candidates_found,
            "collision_risks":  len(self.risks),
            "caution_warnings": len(self.warnings),
            "has_collision_risk": self.has_collision_risk,
            "risks":            self.risks,
            "warnings":         self.warnings,
            "screen_time_ms":   round(self.screen_time_ms, 3),
        }


# ═══════════════════════════════════════════════════════════════════════════
# Spatial Index
# ═══════════════════════════════════════════════════════════════════════════

class SpatialIndex:
    """
    KDTree-backed spatial index over a snapshot of debris positions.

    Build once per telemetry frame, then query for every satellite.
    The tree is immutable after construction — rebuild when positions change.

    Parameters
    ----------
    debris_positions : array-like, shape (N, 3)
        ECI position vectors of all tracked debris objects  [km].
    debris_ids : list[str]
        Registry IDs corresponding to each row in *debris_positions*.
    debris_velocities : array-like, shape (N, 3)
        ECI velocity vectors  [km/s].  Required for TCA computation.
    query_radius_km : float
        Default search radius used by :meth:`find_nearby_objects`.
    """

    def __init__(
        self,
        debris_positions:  np.ndarray,          # (N, 3) [km]
        debris_ids:        List[str],
        debris_velocities: np.ndarray,           # (N, 3) [km/s]
        query_radius_km:   float = DEFAULT_QUERY_RADIUS_KM,
    ) -> None:
        if len(debris_ids) == 0:
            raise ValueError("Cannot build a SpatialIndex with zero debris objects")
        if debris_positions.shape[0] != len(debris_ids):
            raise ValueError("debris_positions row count must match len(debris_ids)")

        self._positions  = np.asarray(debris_positions, dtype=np.float64)
        self._velocities = np.asarray(debris_velocities, dtype=np.float64)
        self._ids        = list(debris_ids)
        self._radius     = query_radius_km
        self._tree       = KDTree(self._positions)   # O(N log N) build
        self._built_at   = time.monotonic()

        logger.debug(
            "SpatialIndex built: %d debris objects, radius=%.1f km",
            len(self._ids), query_radius_km,
        )

    # ─── properties ────────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        """Number of debris objects indexed."""
        return len(self._ids)

    @property
    def query_radius_km(self) -> float:
        return self._radius

    @property
    def age_seconds(self) -> float:
        """Seconds since this index was built."""
        return time.monotonic() - self._built_at

    # ─── public methods ────────────────────────────────────────────────────

    def find_nearby_objects(
        self,
        satellite_position: Union[np.ndarray, Tuple[float, float, float]],
        *,
        radius_km: Optional[float] = None,
    ) -> List[NearbyObject]:
        """
        Phase-1 spatial query: return all debris within *radius_km* of
        the given satellite position.

        Parameters
        ----------
        satellite_position : array-like (3,)
            ECI position of the satellite  [km].
        radius_km : float | None
            Search radius  [km].  Defaults to ``self.query_radius_km``.

        Returns
        -------
        list[NearbyObject]
            Sorted by instantaneous distance (closest first).
        """
        r = radius_km or self._radius
        sat_pos = np.asarray(satellite_position, dtype=np.float64).reshape(3)

        # KDTree.query_ball_point returns indices of all points within radius
        indices = self._tree.query_ball_point(sat_pos, r=r)

        nearby: List[NearbyObject] = []
        for idx in indices:
            deb_pos  = self._positions[idx]
            dist     = float(np.linalg.norm(sat_pos - deb_pos))
            nearby.append(NearbyObject(
                debris_id       = self._ids[idx],
                debris_index    = idx,
                current_dist_km = dist,
                position        = deb_pos.copy(),
                velocity        = self._velocities[idx].copy(),
            ))

        # Sort by instantaneous distance
        nearby.sort(key=lambda o: o.current_dist_km)
        return nearby


# ═══════════════════════════════════════════════════════════════════════════
# Public factory function
# ═══════════════════════════════════════════════════════════════════════════

def build_spatial_index(
    debris_positions: Union[List, np.ndarray],
    *,
    debris_ids:        List[str],
    debris_velocities: Optional[Union[List, np.ndarray]] = None,
    query_radius_km:   float = DEFAULT_QUERY_RADIUS_KM,
) -> SpatialIndex:
    """
    Build a KDTree-backed spatial index from a snapshot of debris positions.

    Parameters
    ----------
    debris_positions : list | ndarray  shape (N, 3)
        ECI positions [km] — one row per debris object.
    debris_ids : list[str]
        Registry ID for each row (same order).
    debris_velocities : list | ndarray  shape (N, 3), optional
        ECI velocities [km/s].  If omitted, zeros are assumed (no TCA calc).
    query_radius_km : float
        Default search radius for ``find_nearby_objects`` calls.

    Returns
    -------
    SpatialIndex
        Ready to query immediately.

    Examples
    --------
    >>> positions = [[7000, 0, 0], [7001, 0, 0]]
    >>> index = build_spatial_index(
    ...     positions,
    ...     debris_ids=["DEB-1", "DEB-2"],
    ...     query_radius_km=50.0,
    ... )
    >>> index.size
    2
    """
    pos_arr = np.asarray(debris_positions, dtype=np.float64)
    if pos_arr.ndim == 1:
        pos_arr = pos_arr.reshape(1, 3)

    if debris_velocities is not None:
        vel_arr = np.asarray(debris_velocities, dtype=np.float64)
        if vel_arr.ndim == 1:
            vel_arr = vel_arr.reshape(1, 3)
    else:
        vel_arr = np.zeros_like(pos_arr)

    return SpatialIndex(
        debris_positions  = pos_arr,
        debris_ids        = debris_ids,
        debris_velocities = vel_arr,
        query_radius_km   = query_radius_km,
    )


# ═══════════════════════════════════════════════════════════════════════════
# TCA computation  (linear look-ahead + ternary search refinement)
# ═══════════════════════════════════════════════════════════════════════════

def _range_at_time(
    sat_pos: np.ndarray,
    sat_vel: np.ndarray,
    deb_pos: np.ndarray,
    deb_vel: np.ndarray,
    t: float,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute range between linearised trajectories at time *t* [s].

    Uses straight-line (constant-velocity) propagation.  Accurate to
    first order over short look-ahead horizons (< ~10 minutes in LEO).

    Returns (range_km, sat_pos_t, deb_pos_t).
    """
    sp = sat_pos + sat_vel * t
    dp = deb_pos + deb_vel * t
    return float(np.linalg.norm(sp - dp)), sp, dp


def _analytical_tca(
    rel_pos: np.ndarray,
    rel_vel: np.ndarray,
    horizon_s: float,
) -> float:
    """
    Closed-form TCA for two objects with constant velocities.

    The squared range  r(t)² = |Δr + Δv · t|²  is a quadratic in t.
    Minimum is at  t* = −(Δr · Δv) / |Δv|²,  clamped to [0, horizon_s].

    Parameters
    ----------
    rel_pos : ndarray (3,)  Δr = r_sat − r_deb  [km]
    rel_vel : ndarray (3,)  Δv = v_sat − v_deb  [km/s]
    horizon_s : float       Maximum look-ahead  [s]

    Returns
    -------
    float  – TCA time [s] in [0, horizon_s]
    """
    dv2 = float(np.dot(rel_vel, rel_vel))
    if dv2 < 1e-20:
        return 0.0
    t_star = -float(np.dot(rel_pos, rel_vel)) / dv2
    return float(np.clip(t_star, 0.0, horizon_s))


def compute_tca(
    sat_pos:   np.ndarray,
    sat_vel:   np.ndarray,
    deb_pos:   np.ndarray,
    deb_vel:   np.ndarray,
    *,
    horizon_s: float = DEFAULT_TCA_HORIZON_S,
    steps:     int   = DEFAULT_TCA_STEPS,
) -> TCAResult:
    """
    Compute the Time of Closest Approach (TCA) between a satellite and a
    debris object over a look-ahead horizon using linear propagation.

    Algorithm
    ---------
    1. **Closed-form estimate**: for constant-velocity motion the TCA is the
       minimum of a quadratic r(t)² — solvable analytically.
    2. **Grid verification**: sample *steps* equally-spaced points in
       ``[0, horizon_s]`` to guard against non-linearities when the
       analytical minimum is outside the horizon bracket.
    3. **Ternary-search refinement**: narrow the bracketed interval around
       the best grid point for sub-second precision.

    Parameters
    ----------
    sat_pos   : ndarray (3,)  – satellite ECI position  [km]
    sat_vel   : ndarray (3,)  – satellite ECI velocity  [km/s]
    deb_pos   : ndarray (3,)  – debris ECI position     [km]
    deb_vel   : ndarray (3,)  – debris ECI velocity     [km/s]
    horizon_s : float         – look-ahead window       [s]
    steps     : int           – grid resolution

    Returns
    -------
    TCAResult
        Contains TCA time, miss distance, collision flag, and risk level.
    """
    rel_pos = sat_pos - deb_pos
    rel_vel = sat_vel - deb_vel

    # ── Step 1: Analytical TCA estimate ─────────────────────────────────────
    t_analytic = _analytical_tca(rel_pos, rel_vel, horizon_s)

    # ── Step 2: Grid scan to find global minimum ─────────────────────────────
    t_grid  = np.linspace(0.0, horizon_s, steps)
    r_grid  = np.linalg.norm(
        rel_pos + rel_vel * t_grid[:, None],   # (steps, 3)
        axis=1,
    )
    best_idx    = int(np.argmin(r_grid))
    t_best_grid = float(t_grid[best_idx])

    # ── Step 3: Ternary-search refinement around the best grid interval ───────
    dt = horizon_s / steps
    lo = max(0.0, t_best_grid - dt)
    hi = min(horizon_s, t_best_grid + dt)

    for _ in range(TERNARY_ITERATIONS):
        m1 = lo + (hi - lo) / 3.0
        m2 = hi - (hi - lo) / 3.0
        r1 = float(np.linalg.norm(rel_pos + rel_vel * m1))
        r2 = float(np.linalg.norm(rel_pos + rel_vel * m2))
        if r1 < r2:
            hi = m2
        else:
            lo = m1

    t_tca = (lo + hi) / 2.0

    # Compare analytical vs refined; take the one with smaller range
    r_analytic  = float(np.linalg.norm(rel_pos + rel_vel * t_analytic))
    r_refined   = float(np.linalg.norm(rel_pos + rel_vel * t_tca))
    if r_analytic < r_refined:
        t_tca = t_analytic

    miss_dist, sp_tca, dp_tca = _range_at_time(sat_pos, sat_vel, deb_pos, deb_vel, t_tca)

    collision_risk = miss_dist < COLLISION_THRESHOLD_KM
    risk_level     = _risk_level(miss_dist)

    return TCAResult(
        tca_s            = round(t_tca, 3),
        miss_distance_km = round(miss_dist, 6),
        collision_risk   = collision_risk,
        sat_pos_at_tca   = sp_tca,
        deb_pos_at_tca   = dp_tca,
        risk_level       = risk_level,
    )


def _risk_level(dist_km: float) -> str:
    """Classify miss distance into a named risk tier."""
    if dist_km < COLLISION_THRESHOLD_KM:   # < 0.1 km
        return "CRITICAL"
    if dist_km < 0.5:
        return "WARNING"
    if dist_km < 2.0:
        return "CAUTION"
    return "NOMINAL"


# ═══════════════════════════════════════════════════════════════════════════
# Convenience pipeline helpers
# ═══════════════════════════════════════════════════════════════════════════

def _sv_to_arrays(
    obj: Union[SimSatellite, SimDebris],
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract ECI position and velocity as numpy arrays from a sim object."""
    pos = np.array([obj.position.x, obj.position.y, obj.position.z], dtype=np.float64)
    vel = np.array([obj.velocity.vx, obj.velocity.vy, obj.velocity.vz], dtype=np.float64)
    return pos, vel


def screen_satellite_vs_debris(
    satellite:   SimSatellite,
    debris_list: List[SimDebris],
    *,
    radius_km:  float = DEFAULT_QUERY_RADIUS_KM,
    horizon_s:  float = DEFAULT_TCA_HORIZON_S,
) -> CollisionScreenResult:
    """
    Full two-phase collision screen for **one satellite** against a list
    of debris objects.

    Phase 1 – KDTree query within *radius_km*.
    Phase 2 – TCA refinement for every candidate.

    Returns a :class:`CollisionScreenResult` with collision risks and
    caution warnings.

    Parameters
    ----------
    satellite   : SimSatellite
    debris_list : list[SimDebris]
    radius_km   : float – spatial pre-filter radius  [km]
    horizon_s   : float – TCA look-ahead window      [s]
    """
    t0 = time.perf_counter()

    # ── Guard: empty debris list ─────────────────────────────────────────────
    if not debris_list:
        return CollisionScreenResult(
            satellite_id    = satellite.id,
            satellite_name  = satellite.name,
            query_radius_km = radius_km,
            candidates_found = 0,
            risks    = [],
            warnings = [],
            screen_time_ms = 0.0,
        )

    # ── Build position / velocity arrays ────────────────────────────────────
    pos_arr = np.array([[d.position.x, d.position.y, d.position.z] for d in debris_list])
    vel_arr = np.array([[d.velocity.vx, d.velocity.vy, d.velocity.vz] for d in debris_list])
    ids     = [d.id for d in debris_list]

    # ── Phase 1: KDTree spatial pre-filter ──────────────────────────────────
    index    = build_spatial_index(pos_arr, debris_ids=ids, debris_velocities=vel_arr,
                                   query_radius_km=radius_km)
    sat_pos, sat_vel = _sv_to_arrays(satellite)
    candidates = index.find_nearby_objects(sat_pos)

    # ── Phase 2: TCA refinement ──────────────────────────────────────────────
    risks:    List[dict] = []
    warnings: List[dict] = []

    for cand in candidates:
        tca = compute_tca(
            sat_pos, sat_vel,
            cand.position, cand.velocity,
            horizon_s = horizon_s,
        )
        entry = {
            "debris_id":        cand.debris_id,
            "current_dist_km":  round(cand.current_dist_km, 4),
            **tca.to_dict(),
        }
        if tca.collision_risk:
            risks.append(entry)
        elif tca.risk_level in ("WARNING", "CAUTION"):
            warnings.append(entry)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    if risks:
        logger.warning(
            "COLLISION RISK  sat=%s  %d critical pair(s)  screen_time=%.2f ms",
            satellite.name, len(risks), elapsed_ms,
        )
    elif warnings:
        logger.info(
            "CAUTION  sat=%s  %d warning pair(s)  screen_time=%.2f ms",
            satellite.name, len(warnings), elapsed_ms,
        )

    return CollisionScreenResult(
        satellite_id     = satellite.id,
        satellite_name   = satellite.name,
        query_radius_km  = radius_km,
        candidates_found = len(candidates),
        risks            = risks,
        warnings         = warnings,
        screen_time_ms   = elapsed_ms,
    )


def screen_all(
    *,
    query_radius_km: float = DEFAULT_QUERY_RADIUS_KM,
    tca_horizon_s:   float = DEFAULT_TCA_HORIZON_S,
) -> List[CollisionScreenResult]:
    """
    Screen **every satellite** in the registry against **every debris object**
    in the registry.

    This is the top-level convenience function that pulls objects directly
    from the in-memory registries (``app.data.registry``).

    Returns
    -------
    list[CollisionScreenResult]
        One result per satellite.  Satellites with no nearby debris have
        ``candidates_found == 0``.

    Notes
    -----
    The debris KDTree is built **once** and shared across all satellite
    queries, so the total cost is:
        O(N log N)  [tree build]  +  O(S · k · log N)  [S satellite queries]
    where N = # debris, S = # satellites, k = avg candidates per satellite.
    """
    from app.data.registry import satellites, debris as deb_registry

    sat_list = list(satellites.values())
    deb_list = list(deb_registry.values())

    if not sat_list:
        logger.info("screen_all: no satellites in registry")
        return []
    if not deb_list:
        logger.info("screen_all: no debris in registry – nothing to screen")
        return [
            CollisionScreenResult(
                satellite_id     = s.id,
                satellite_name   = s.name,
                query_radius_km  = query_radius_km,
                candidates_found = 0,
                risks    = [],
                warnings = [],
                screen_time_ms = 0.0,
            )
            for s in sat_list
        ]

    # Build the single shared KDTree over all debris
    pos_arr = np.array([[d.position.x, d.position.y, d.position.z] for d in deb_list])
    vel_arr = np.array([[d.velocity.vx, d.velocity.vy, d.velocity.vz] for d in deb_list])
    ids     = [d.id for d in deb_list]

    shared_index = build_spatial_index(
        pos_arr, debris_ids=ids, debris_velocities=vel_arr,
        query_radius_km=query_radius_km,
    )

    results: List[CollisionScreenResult] = []
    t_global = time.perf_counter()

    for sat in sat_list:
        t0 = time.perf_counter()
        sat_pos, sat_vel = _sv_to_arrays(sat)
        candidates = shared_index.find_nearby_objects(sat_pos)

        risks:    List[dict] = []
        warnings: List[dict] = []

        for cand in candidates:
            tca = compute_tca(
                sat_pos, sat_vel,
                cand.position, cand.velocity,
                horizon_s = tca_horizon_s,
            )
            entry = {
                "debris_id":       cand.debris_id,
                "current_dist_km": round(cand.current_dist_km, 4),
                **tca.to_dict(),
            }
            if tca.collision_risk:
                risks.append(entry)
            elif tca.risk_level in ("WARNING", "CAUTION"):
                warnings.append(entry)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        results.append(CollisionScreenResult(
            satellite_id     = sat.id,
            satellite_name   = sat.name,
            query_radius_km  = query_radius_km,
            candidates_found = len(candidates),
            risks            = risks,
            warnings         = warnings,
            screen_time_ms   = elapsed_ms,
        ))

    total_ms = (time.perf_counter() - t_global) * 1000.0
    total_risks = sum(len(r.risks) for r in results)
    logger.info(
        "screen_all: %d sats × %d debris  total_risks=%d  %.2f ms",
        len(sat_list), len(deb_list), total_risks, total_ms,
    )
    return results
