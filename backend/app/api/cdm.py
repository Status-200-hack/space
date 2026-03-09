"""
CDM (Conjunction Data Message) API
====================================
Endpoints for running 24-hour conjunction detection and querying results.

Endpoints
---------
  POST /api/v1/cdm/run
      Trigger a full conjunction detection run against all registry objects.

  POST /api/v1/cdm/run-custom
      Supply satellite and debris lists inline (no registry lookup).

  GET  /api/v1/cdm/summary
      Last-run statistics and most critical CDM.

  GET  /api/v1/cdm/
      List all CDMs from the last run, with optional risk/ID filters.

  GET  /api/v1/cdm/{satellite_id}
      All CDMs involving a specific satellite.

  GET  /api/v1/cdm/config
      Current service configuration (horizon, step, radius, threshold).

  POST /api/v1/cdm/configure
      Reconfigure the shared service instance.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel, Field

from app.services.conjunction_service import (
    get_service,
    reset_service,
    ConjunctionDataMessage,
    HORIZON_HOURS, STEP_SECONDS, SCREEN_RADIUS_KM, CDM_THRESHOLD_KM,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/cdm")


# ═══════════════════════════════════════════════════════════════════════════
# Request / Response schemas
# ═══════════════════════════════════════════════════════════════════════════

class RunRequest(BaseModel):
    horizon_hours:    float = Field(HORIZON_HOURS, gt=0, le=72,
                                     description="Propagation horizon [h] (max 72)")
    step_seconds:     float = Field(STEP_SECONDS,  gt=0, le=300,
                                     description="RK4 integration step [s]")
    screen_radius_km: float = Field(SCREEN_RADIUS_KM, gt=0,
                                     description="KDTree spatial pre-filter radius [km]")
    cdm_threshold_km: float = Field(CDM_THRESHOLD_KM, gt=0,
                                     description="Only emit CDMs below this miss distance [km]")


class ConfigureRequest(BaseModel):
    horizon_hours:    float = Field(HORIZON_HOURS, gt=0, le=72)
    step_seconds:     float = Field(STEP_SECONDS,  gt=0, le=300)
    screen_radius_km: float = Field(SCREEN_RADIUS_KM, gt=0)
    cdm_threshold_km: float = Field(CDM_THRESHOLD_KM, gt=0)


# ═══════════════════════════════════════════════════════════════════════════
# Endpoints
# ═══════════════════════════════════════════════════════════════════════════

@router.get("/config", summary="Current conjunction service configuration")
async def get_config():
    svc = get_service()
    return {
        "horizon_hours":    svc.horizon_hours,
        "step_seconds":     svc.step_seconds,
        "screen_radius_km": svc.screen_radius_km,
        "cdm_threshold_km": svc.cdm_threshold_km,
    }


@router.post("/configure", summary="Reconfigure the shared conjunction service")
async def configure(req: ConfigureRequest):
    svc = reset_service(
        horizon_hours    = req.horizon_hours,
        step_seconds     = req.step_seconds,
        screen_radius_km = req.screen_radius_km,
        cdm_threshold_km = req.cdm_threshold_km,
    )
    return {"status": "reconfigured", **req.model_dump()}


@router.post(
    "/run",
    summary="Run 24-hour conjunction detection against registry objects",
)
async def run_conjunction_detection(req: RunRequest):
    """
    Trigger a full conjunction detection pass against all satellites and debris
    objects currently in the in-memory registries.

    The run propagates every object's trajectory (RK4, J2, drag) for the
    configured horizon, screens pairs with a KDTree at every time step, and
    refines TCA for each candidate window.

    **Warning:** For large constellations this can take several seconds.
    Use the ``/cdm/run-background`` endpoint for async execution.
    """
    from app.data.registry import satellites, debris as deb_registry

    sat_count = len(satellites)
    deb_count = len(deb_registry)

    if sat_count == 0:
        raise HTTPException(400, "No satellites in registry. Register at least one satellite first.")
    if deb_count == 0:
        raise HTTPException(400, "No debris objects in registry. Register at least one debris object first.")

    svc = get_service()
    svc.horizon_hours     = req.horizon_hours
    svc.step_seconds      = req.step_seconds
    svc.screen_radius_km  = req.screen_radius_km
    svc.cdm_threshold_km  = req.cdm_threshold_km

    cdms = svc.run_from_registry()
    summary = svc.get_summary()

    return {
        "conjunctions_detected": len(cdms),
        "summary": summary,
        "cdms": [c.to_dict() for c in cdms],
    }


@router.post(
    "/run-background",
    status_code=202,
    summary="Launch conjunction detection as a background task",
)
async def run_background(req: RunRequest, background_tasks: BackgroundTasks):
    """Launch the detection run asynchronously. Poll ``/cdm/summary`` for results."""
    from app.data.registry import satellites, debris as deb_registry

    if not satellites:
        raise HTTPException(400, "No satellites in registry.")
    if not deb_registry:
        raise HTTPException(400, "No debris in registry.")

    svc = get_service()
    svc.horizon_hours     = req.horizon_hours
    svc.step_seconds      = req.step_seconds
    svc.screen_radius_km  = req.screen_radius_km
    svc.cdm_threshold_km  = req.cdm_threshold_km

    def _run():
        svc.run_from_registry()

    background_tasks.add_task(_run)
    return {"status": "accepted", "message": "Conjunction detection started in background. Poll /cdm/summary for results."}


@router.get(
    "/summary",
    summary="Summary of the last conjunction detection run",
)
async def get_summary():
    return get_service().get_summary()


@router.get(
    "/",
    summary="List CDMs from the last run",
)
async def list_cdms(
    risk_level:   Optional[str] = Query(None, description="Filter by risk level: CRITICAL|WARNING|CAUTION|NOMINAL"),
    satellite_id: Optional[str] = Query(None, description="Filter by satellite ID"),
    debris_id:    Optional[str] = Query(None, description="Filter by debris ID"),
    limit:        int            = Query(200, ge=1, le=2000),
):
    """
    Return CDMs from the most recent conjunction run, sorted by miss distance
    (closest approach first).  Optionally filter by risk level or object ID.
    """
    svc = get_service()
    if not svc.last_run_at:
        return {"status": "no_run_yet", "cdms": [], "count": 0}

    cdms = svc.get_cdms(
        risk_level   = risk_level,
        satellite_id = satellite_id,
        debris_id    = debris_id,
        limit        = limit,
    )
    return {
        "count":      len(cdms),
        "run_at":     svc.last_run_at.isoformat() if svc.last_run_at else None,
        "cdms":       [c.to_dict() for c in cdms],
    }


@router.get(
    "/{satellite_id}",
    summary="All CDMs involving a specific satellite",
)
async def cdms_for_satellite(
    satellite_id: str,
    risk_level:   Optional[str] = Query(None),
    limit:        int            = Query(100, ge=1, le=1000),
):
    svc = get_service()
    if not svc.last_run_at:
        raise HTTPException(404, "No conjunction run has been executed yet.")

    cdms = svc.get_cdms(satellite_id=satellite_id, risk_level=risk_level, limit=limit)
    if not cdms:
        return {"satellite_id": satellite_id, "count": 0, "cdms": []}
    return {"satellite_id": satellite_id, "count": len(cdms), "cdms": [c.to_dict() for c in cdms]}
