"""
Registry API – CRUD for the in-memory simulation object registries.

Exposes endpoints to:
  • Create / read / list / delete SimSatellite entries
  • Create / read / list / delete SimDebris entries
  • Ingest TelemetryRequest frames (update both registries atomically)
  • Query registry statistics
"""

from __future__ import annotations

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query

from app.models.sim_objects import (
    SimSatellite,
    SimDebris,
    StateVector,
    SatelliteStatus,
    TelemetryRequest,
)
from app.data.registry import (
    # satellites
    add_satellite, get_satellite, remove_satellite,
    list_satellites, update_satellite_state,
    # debris
    add_debris, get_debris, remove_debris,
    list_debris, update_debris_state,
    # cross-registry
    registry_stats, apply_telemetry, clear_all,
)

router = APIRouter(prefix="/registry")


# ═══════════════════════════════════════════════════════════════════════════
# Registry stats & housekeeping
# ═══════════════════════════════════════════════════════════════════════════

@router.get("/stats", summary="Registry snapshot statistics")
async def stats():
    """Return total counts of satellites and debris in the live registries."""
    return registry_stats()


@router.delete("/all", status_code=204, summary="Clear all registries (simulation reset)")
async def clear_registries():
    """Wipe all registered objects. Use with care – intended for simulation resets."""
    clear_all()


# ═══════════════════════════════════════════════════════════════════════════
# Satellite registry
# ═══════════════════════════════════════════════════════════════════════════

@router.post(
    "/satellites",
    response_model=SimSatellite,
    status_code=201,
    summary="Register a satellite in the simulation registry",
)
async def register_satellite(payload: SimSatellite):
    return add_satellite(payload)


@router.get(
    "/satellites",
    response_model=List[SimSatellite],
    summary="List simulation satellites",
)
async def get_satellites(
    status: Optional[SatelliteStatus] = Query(None),
    limit: int = Query(500, ge=1, le=5000),
    offset: int = Query(0, ge=0),
):
    return list_satellites(status=status, limit=limit, offset=offset)


@router.get(
    "/satellites/{satellite_id}",
    response_model=SimSatellite,
    summary="Get simulation satellite by ID",
)
async def get_satellite_by_id(satellite_id: str):
    sat = get_satellite(satellite_id)
    if not sat:
        raise HTTPException(404, f"Satellite '{satellite_id}' not found in registry")
    return sat


@router.delete(
    "/satellites/{satellite_id}",
    status_code=204,
    summary="Remove satellite from registry",
)
async def delete_satellite(satellite_id: str):
    if not remove_satellite(satellite_id):
        raise HTTPException(404, f"Satellite '{satellite_id}' not found in registry")


# ═══════════════════════════════════════════════════════════════════════════
# Debris registry
# ═══════════════════════════════════════════════════════════════════════════

@router.post(
    "/debris",
    response_model=SimDebris,
    status_code=201,
    summary="Register a debris object in the simulation registry",
)
async def register_debris(payload: SimDebris):
    return add_debris(payload)


@router.get(
    "/debris",
    response_model=List[SimDebris],
    summary="List simulation debris objects",
)
async def get_debris_list(
    limit: int = Query(500, ge=1, le=5000),
    offset: int = Query(0, ge=0),
):
    return list_debris(limit=limit, offset=offset)


@router.get(
    "/debris/{debris_id}",
    response_model=SimDebris,
    summary="Get debris object by ID",
)
async def get_debris_by_id(debris_id: str):
    obj = get_debris(debris_id)
    if not obj:
        raise HTTPException(404, f"Debris '{debris_id}' not found in registry")
    return obj


@router.delete(
    "/debris/{debris_id}",
    status_code=204,
    summary="Remove debris from registry",
)
async def delete_debris(debris_id: str):
    if not remove_debris(debris_id):
        raise HTTPException(404, f"Debris '{debris_id}' not found in registry")


# ═══════════════════════════════════════════════════════════════════════════
# Telemetry ingest
# ═══════════════════════════════════════════════════════════════════════════

@router.post(
    "/telemetry",
    summary="Ingest a telemetry frame into the registries",
)
async def ingest_telemetry(request: TelemetryRequest):
    """
    Apply a TelemetryRequest to both registries atomically.
    Objects not found in either registry are reported in `unknown_ids`.
    """
    return apply_telemetry(request)
