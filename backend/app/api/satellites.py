"""
Satellites API – CRUD + propagation endpoints.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional

from app.models.satellite import Satellite, SatelliteCreate, SatelliteUpdate, SatelliteStatus
from app.models.orbital_state import OrbitalState
from app.services.satellite_service import SatelliteService

router = APIRouter(prefix="/satellites")
_svc = SatelliteService()


@router.post("", response_model=Satellite, status_code=201, summary="Register a new satellite")
async def create_satellite(payload: SatelliteCreate):
    return _svc.create(payload)


@router.get("", response_model=List[Satellite], summary="List all satellites")
async def list_satellites(
    constellation: Optional[str] = Query(None),
    status: Optional[SatelliteStatus] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    return _svc.list_all(constellation=constellation, status=status, limit=limit, offset=offset)


@router.get("/count", summary="Satellite count")
async def count_satellites():
    return {"count": _svc.count()}


@router.get("/{satellite_id}", response_model=Satellite, summary="Get satellite by ID")
async def get_satellite(satellite_id: str):
    sat = _svc.get(satellite_id)
    if not sat:
        raise HTTPException(404, f"Satellite '{satellite_id}' not found")
    return sat


@router.patch("/{satellite_id}", response_model=Satellite, summary="Update satellite")
async def update_satellite(satellite_id: str, payload: SatelliteUpdate):
    sat = _svc.update(satellite_id, payload)
    if not sat:
        raise HTTPException(404, f"Satellite '{satellite_id}' not found")
    return sat


@router.delete("/{satellite_id}", status_code=204, summary="Delete satellite")
async def delete_satellite(satellite_id: str):
    if not _svc.delete(satellite_id):
        raise HTTPException(404, f"Satellite '{satellite_id}' not found")


@router.get("/{satellite_id}/state", response_model=OrbitalState, summary="Current orbital state")
async def current_state(satellite_id: str):
    try:
        return _svc.current_state(satellite_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc))


@router.get(
    "/{satellite_id}/propagate",
    response_model=List[OrbitalState],
    summary="Propagate orbit",
)
async def propagate(
    satellite_id: str,
    duration_seconds: float = Query(3600.0, gt=0, le=604800, description="Propagation duration [s]"),
):
    try:
        return _svc.propagate(satellite_id, duration_seconds)
    except KeyError as exc:
        raise HTTPException(404, str(exc))
