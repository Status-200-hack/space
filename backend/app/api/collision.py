"""
Collision Screening API
=======================

Exposes the KDTree-backed spatial collision detection engine via REST.

Endpoints
---------
  POST /api/v1/collision/index
      Build a spatial index from an explicit list of debris positions and
      query it for a given satellite position.

  POST /api/v1/collision/screen/{satellite_id}
      Run a full two-phase screen (KDTree + TCA) for one registry satellite
      against all registry debris objects.

  POST /api/v1/collision/screen-all
      Screen every satellite in the registry against every debris object
      using a single shared KDTree.

  GET  /api/v1/collision/thresholds
      Return the configured distance thresholds and risk levels.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.models.sim_objects import StateVector
from app.physics.spatial_index import (
    build_spatial_index,
    compute_tca,
    screen_satellite_vs_debris,
    screen_all,
    COLLISION_THRESHOLD_KM,
    DEFAULT_QUERY_RADIUS_KM,
    DEFAULT_TCA_HORIZON_S,
)
from app.data.registry import get_satellite, get_debris, list_debris

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/collision")


# ═══════════════════════════════════════════════════════════════════════════
# Request/Response schemas
# ═══════════════════════════════════════════════════════════════════════════

class DebrisEntry(BaseModel):
    """A single debris object with its ECI state for ad-hoc index queries."""
    debris_id:  str
    position:   StateVector
    velocity:   StateVector


class IndexQueryRequest(BaseModel):
    """
    Request body for POST /collision/index.

    Build a fresh KDTree from the supplied debris list and immediately
    query it for the given satellite position.
    """
    satellite_position: StateVector
    satellite_velocity: StateVector
    debris_objects:     List[DebrisEntry] = Field(..., min_length=1)
    query_radius_km:    float  = Field(DEFAULT_QUERY_RADIUS_KM, gt=0)
    tca_horizon_s:      float  = Field(DEFAULT_TCA_HORIZON_S,   gt=0)


class ScreenRequest(BaseModel):
    """Parameters for a registry-based screen operation."""
    query_radius_km:  float = Field(DEFAULT_QUERY_RADIUS_KM, gt=0,
                                    description="Spatial pre-filter radius [km]")
    tca_horizon_s:    float = Field(DEFAULT_TCA_HORIZON_S,   gt=0,
                                    description="TCA look-ahead window [s]")


# ═══════════════════════════════════════════════════════════════════════════
# Endpoints
# ═══════════════════════════════════════════════════════════════════════════

@router.get(
    "/thresholds",
    summary="Collision detection thresholds and risk levels",
)
async def get_thresholds():
    """Return the distance thresholds that define each risk level."""
    return {
        "collision_threshold_km":    COLLISION_THRESHOLD_KM,
        "default_query_radius_km":   DEFAULT_QUERY_RADIUS_KM,
        "default_tca_horizon_s":     DEFAULT_TCA_HORIZON_S,
        "risk_levels": [
            {"level": "CRITICAL", "condition": f"miss_distance < {COLLISION_THRESHOLD_KM} km (collision risk)"},
            {"level": "WARNING",  "condition": "miss_distance < 0.5 km"},
            {"level": "CAUTION",  "condition": "miss_distance < 2.0 km"},
            {"level": "NOMINAL",  "condition": "miss_distance >= 2.0 km"},
        ],
    }


@router.post(
    "/index",
    summary="Build spatial index from payload and query for nearby debris",
)
async def query_index(req: IndexQueryRequest):
    """
    Build a KDTree from the *debris_objects* in the request body and
    run a Phase-1 + Phase-2 screen for the supplied satellite state.

    Useful for visualisation, simulation playback, or testing without
    needing any objects registered in the in-memory registry.
    """
    import numpy as np

    pos_arr = np.array(
        [[d.position.x, d.position.y, d.position.z] for d in req.debris_objects]
    )
    vel_arr = np.array(
        [[d.velocity.vx, d.velocity.vy, d.velocity.vz] for d in req.debris_objects]
    )
    ids = [d.debris_id for d in req.debris_objects]

    index = build_spatial_index(
        pos_arr, debris_ids=ids, debris_velocities=vel_arr,
        query_radius_km=req.query_radius_km,
    )

    sat_pos = np.array([req.satellite_position.x, req.satellite_position.y, req.satellite_position.z])
    sat_vel = np.array([req.satellite_velocity.vx, req.satellite_velocity.vy, req.satellite_velocity.vz])

    candidates = index.find_nearby_objects(sat_pos)

    results = []
    for cand in candidates:
        tca = compute_tca(
            sat_pos, sat_vel,
            cand.position, cand.velocity,
            horizon_s=req.tca_horizon_s,
        )
        results.append({
            "debris_id":        cand.debris_id,
            "current_dist_km":  round(cand.current_dist_km, 4),
            **tca.to_dict(),
        })

    return {
        "index_size":        index.size,
        "query_radius_km":   req.query_radius_km,
        "candidates_found":  len(candidates),
        "tca_results":       results,
        "collision_risks":   sum(1 for r in results if r["collision_risk"]),
    }


@router.post(
    "/screen/{satellite_id}",
    summary="Screen one registry satellite against all registry debris",
)
async def screen_one_satellite(satellite_id: str, req: ScreenRequest):
    """
    Look up *satellite_id* in the registry, build a KDTree from all
    registered debris objects, and run the full two-phase collision screen.

    Returns detailed TCA results for every debris object within
    *query_radius_km*, flagging those below the collision threshold.
    """
    sat = get_satellite(satellite_id)
    if not sat:
        raise HTTPException(404, f"Satellite '{satellite_id}' not found in registry")

    deb_list = list_debris(limit=10_000)
    if not deb_list:
        return {
            "satellite_id":     satellite_id,
            "satellite_name":   sat.name,
            "message":          "No debris objects in registry",
            "candidates_found": 0,
            "collision_risks":  0,
        }

    result = screen_satellite_vs_debris(
        sat, deb_list,
        radius_km = req.query_radius_km,
        horizon_s = req.tca_horizon_s,
    )
    return result.to_dict()


@router.post(
    "/screen-all",
    summary="Screen every satellite in the registry against all debris",
)
async def screen_all_satellites(req: ScreenRequest):
    """
    Full constellation collision screen using a single shared KDTree.

    The debris tree is built once (O(N log N)) and all satellite queries
    run against it, making the total cost O(N log N + S · k · log N)
    instead of O(S · N).

    Large constellations with many debris objects should run this as an
    async background task in production.
    """
    results = screen_all(
        query_radius_km=req.query_radius_km,
        tca_horizon_s  =req.tca_horizon_s,
    )

    total_risks    = sum(len(r.risks)    for r in results)
    total_warnings = sum(len(r.warnings) for r in results)

    return {
        "satellites_screened": len(results),
        "total_collision_risks":   total_risks,
        "total_caution_warnings":  total_warnings,
        "results": [r.to_dict() for r in results],
    }
