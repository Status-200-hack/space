"""
Maneuvers API – plan, approve, cancel, and monitor orbital maneuvers.
Also exposes POST /schedule for validated maneuver sequence scheduling.
"""

from datetime import datetime
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional

from app.models.maneuver import ManeuverPlan, ManeuverCreate, ManeuverStatus
from app.services.maneuver_service import ManeuverService
from app.services.maneuver_scheduler import get_scheduler, MIN_LATENCY_S, COOLDOWN_S

router = APIRouter(prefix="/maneuvers")
_svc = ManeuverService()


@router.post("", response_model=ManeuverPlan, status_code=201, summary="Create maneuver plan")
async def create_maneuver(payload: ManeuverCreate):
    try:
        return _svc.plan(payload)
    except ValueError as exc:
        raise HTTPException(422, str(exc))


@router.get("", response_model=List[ManeuverPlan], summary="List maneuver plans")
async def list_maneuvers(
    satellite_id: Optional[str] = Query(None),
    status: Optional[ManeuverStatus] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    return _svc.list_all(satellite_id=satellite_id, status=status, limit=limit, offset=offset)


@router.get("/{maneuver_id}", response_model=ManeuverPlan, summary="Get maneuver plan")
async def get_maneuver(maneuver_id: str):
    plan = _svc.get(maneuver_id)
    if not plan:
        raise HTTPException(404, f"Maneuver '{maneuver_id}' not found")
    return plan


@router.post("/{maneuver_id}/approve", response_model=ManeuverPlan, summary="Approve maneuver")
async def approve_maneuver(maneuver_id: str):
    plan = _svc.approve(maneuver_id)
    if not plan:
        raise HTTPException(404, f"Maneuver '{maneuver_id}' not found")
    return plan


@router.post("/{maneuver_id}/cancel", response_model=ManeuverPlan, summary="Cancel maneuver")
async def cancel_maneuver(maneuver_id: str):
    plan = _svc.cancel(maneuver_id)
    if not plan:
        raise HTTPException(404, f"Maneuver '{maneuver_id}' not found")
    return plan


@router.post("/{maneuver_id}/complete", response_model=ManeuverPlan, summary="Mark maneuver as completed")
async def complete_maneuver(maneuver_id: str):
    plan = _svc.mark_completed(maneuver_id)
    if not plan:
        raise HTTPException(404, f"Maneuver '{maneuver_id}' not found")
    return plan


@router.post("/{maneuver_id}/fail", response_model=ManeuverPlan, summary="Mark maneuver as failed")
async def fail_maneuver(maneuver_id: str):
    plan = _svc.mark_failed(maneuver_id)
    if not plan:
        raise HTTPException(404, f"Maneuver '{maneuver_id}' not found")
    return plan


# ════════════════════════════════════════════════════════════════════════
# Maneuver sequence scheduling
# ════════════════════════════════════════════════════════════════════════

class DeltaVVector(BaseModel):
    x: float = Field(0.0, description="ECI ΔV x-component [km/s]")
    y: float = Field(0.0, description="ECI ΔV y-component [km/s]")
    z: float = Field(0.0, description="ECI ΔV z-component [km/s]")


class ManeuverItem(BaseModel):
    burn_id:       str          = Field(...,  description="Unique burn identifier")
    burnTime:      str          = Field(...,  description="ISO8601 UTC burn epoch")
    deltaV_vector: DeltaVVector = Field(default_factory=DeltaVVector)


class ScheduleRequest(BaseModel):
    satelliteId:       str               = Field(..., description="Target satellite registry ID")
    maneuver_sequence: List[ManeuverItem] = Field(..., min_length=1, description="Ordered list of burns to schedule")


@router.post(
    "/schedule",
    summary="Schedule a validated maneuver sequence",
    status_code=200,
)
async def schedule_maneuvers(req: ScheduleRequest):
    """
    Validate and schedule a sequence of burns for a satellite.

    Each burn in ``maneuver_sequence`` is checked for:

    | Check             | Rule                                              |
    |-------------------|---------------------------------------------------|
    | **Latency**       | ``burnTime`` ≥ now + 10 s                         |
    | **Cooldown**      | ≥ 600 s gap between consecutive burns             |
    | **Fuel**          | Tsiolkovsky simulation through whole sequence     |
    | **Ground LOS**    | Satellite visible from at least one ground station |

    Burns are accepted in order; the first failure freezes the remaining
    fuel budget, marking subsequent overspend burns **INVALID**.

    Returns the overall status (``SCHEDULED`` | ``PARTIAL`` | ``REJECTED``)
    and a per-burn breakdown including projected mass remaining.
    """
    sequence = [
        {
            "burn_id":      item.burn_id,
            "burnTime":     item.burnTime,
            "deltaV_vector": item.deltaV_vector.model_dump(),
        }
        for item in req.maneuver_sequence
    ]
    try:
        result = get_scheduler().schedule(
            satellite_id      = req.satelliteId,
            maneuver_sequence = sequence,
            epoch             = datetime.utcnow(),
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc))

    return result


@router.get(
    "/schedule/{satellite_id}",
    summary="View scheduled burns for a satellite",
)
async def get_schedule(satellite_id: str):
    """Return all stored (scheduled, invalid, or cancelled) burns for a satellite."""
    burns = get_scheduler().get_schedule(satellite_id)
    return {"satellite_id": satellite_id, "count": len(burns), "burns": burns}


@router.delete(
    "/schedule/{satellite_id}",
    status_code=204,
    summary="Clear all scheduled burns for a satellite",
)
async def clear_schedule(satellite_id: str):
    """Remove all stored schedule entries for the satellite."""
    get_scheduler().clear_schedule(satellite_id)


@router.delete(
    "/schedule/{satellite_id}/{burn_id}",
    summary="Cancel a specific scheduled burn",
)
async def cancel_scheduled_burn(satellite_id: str, burn_id: str):
    """Mark a specific scheduled burn as CANCELLED."""
    cancelled = get_scheduler().cancel_burn(satellite_id, burn_id)
    if not cancelled:
        raise HTTPException(404, f"No active scheduled burn '{burn_id}' for satellite '{satellite_id}'")
    return {"status": "CANCELLED", "burn_id": burn_id, "satellite_id": satellite_id}


@router.get(
    "/schedule-config",
    summary="Scheduling constraint constants",
)
async def schedule_config():
    """Return the timing and fuel constraints used by the scheduler."""
    return {
        "min_latency_s":    MIN_LATENCY_S,
        "cooldown_s":       COOLDOWN_S,
        "description": {
            "min_latency_s": "Minimum seconds from now until burn_time (command propagation delay)",
            "cooldown_s":    "Minimum gap between consecutive burns for the same satellite",
        },
    }
