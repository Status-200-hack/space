"""
Debris API – CRUD and risk-level queries.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional

from app.models.debris import DebrisObject, DebrisCreate, DebrisRiskLevel
from app.services.debris_service import DebrisService

router = APIRouter(prefix="/debris")
_svc = DebrisService()


@router.post("", response_model=DebrisObject, status_code=201, summary="Register debris object")
async def create_debris(payload: DebrisCreate):
    return _svc.create(payload)


@router.get("", response_model=List[DebrisObject], summary="List debris objects")
async def list_debris(
    risk_level: Optional[DebrisRiskLevel] = Query(None),
    tracked_only: bool = Query(True),
    limit: int = Query(200, ge=1, le=2000),
    offset: int = Query(0, ge=0),
):
    return _svc.list_all(risk_level=risk_level, tracked_only=tracked_only, limit=limit, offset=offset)


@router.get("/count", summary="Debris count")
async def count_debris():
    return {"count": _svc.count()}


@router.get("/{debris_id}", response_model=DebrisObject, summary="Get debris object by ID")
async def get_debris(debris_id: str):
    obj = _svc.get(debris_id)
    if not obj:
        raise HTTPException(404, f"Debris object '{debris_id}' not found")
    return obj


@router.delete("/{debris_id}", status_code=204, summary="Delete debris object")
async def delete_debris(debris_id: str):
    if not _svc.delete(debris_id):
        raise HTTPException(404, f"Debris object '{debris_id}' not found")
