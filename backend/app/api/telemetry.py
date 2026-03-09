"""
Telemetry Ingestion API
=======================

Primary endpoint
----------------
  POST /api/v1/telemetry
      Accept a TelemetryFrame containing any mix of SATELLITE and DEBRIS
      state updates, apply them to the in-memory registries at high speed,
      run a CDM (Conjunction Data Message) screen on the affected pairs,
      and return an ACK with processed count and active warning count.

Secondary endpoints (historical / subsystem telemetry)
------------------------------------------------------
  POST /api/v1/telemetry/record          – ingest a single subsystem record
  POST /api/v1/telemetry/record/batch    – bulk subsystem record ingest
  GET  /api/v1/telemetry/{satellite_id}  – query historical records
  GET  /api/v1/telemetry/{satellite_id}/latest
  GET  /api/v1/telemetry/{satellite_id}/stats

High-frequency design notes
----------------------------
• The hot path (POST /api/v1/telemetry) uses fast_update_satellite /
  fast_update_debris which mutate Pydantic model fields in-place via
  object.__setattr__, skipping the model_dump/reconstruct overhead.
• CDM screening runs only over the pairs that involve objects whose state
  was *actually updated* in the current frame – not the full N² grid.
• The response is a plain dict, not a Pydantic response_model, to avoid
  serialisation overhead on every call.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from app.models.sim_objects import ObjectType, TelemetryRequest, StateVector
from app.models.telemetry import TelemetryRecord, TelemetryBatch
from app.services.telemetry_service import TelemetryService
from app.services.cdm_service import screen_updated_objects, active_warning_count
from app.data.registry import fast_update_satellite, fast_update_debris

logger = logging.getLogger(__name__)

# ── Shared service for historical / subsystem telemetry records ──────────────
_svc = TelemetryService()

router = APIRouter(prefix="/telemetry")


# ═══════════════════════════════════════════════════════════════════════════
# Response schema
# ═══════════════════════════════════════════════════════════════════════════

class TelemetryACK(BaseModel):
    """
    Canonical ACK returned by POST /api/v1/telemetry.

    Fields
    ------
    status
        Always "ACK" on success.
    processed_count
        Number of objects whose state was successfully written to the registry.
    active_cdm_warnings
        Number of conjunction warnings currently active across the whole
        registry (not just objects in this frame).
    frame_timestamp
        The epoch of the ingested frame (echoed from the request).
    processing_time_ms
        Server-side wall-clock time spent processing this frame [ms].
    unknown_ids
        Object IDs present in the frame that could not be found in either
        registry (position/velocity NOT updated for these).
    """

    status:              str   = "ACK"
    processed_count:     int
    active_cdm_warnings: int
    frame_timestamp:     str
    processing_time_ms:  float
    unknown_ids:         List[str] = Field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════
# Primary telemetry ingest endpoint
# ═══════════════════════════════════════════════════════════════════════════

@router.post(
    "",
    response_model=TelemetryACK,
    summary="Ingest telemetry frame (primary high-frequency endpoint)",
    description=(
        "Accept a mixed frame of SATELLITE and DEBRIS state updates. "
        "Applies position/velocity/fuel changes to the in-memory registries "
        "and runs a fast CDM conjunction screen on affected object pairs. "
        "Designed for high-frequency ingestion (hundreds of updates per second)."
    ),
)
async def ingest_telemetry_frame(frame: TelemetryRequest) -> TelemetryACK:
    t0 = time.perf_counter()

    processed = 0
    unknown_ids: List[str] = []
    updated: List[tuple[str, str]] = []   # (object_id, object_type) for CDM

    for entry in frame.objects:
        oid  = entry.object_id
        sv   = entry.state

        if entry.object_type == ObjectType.SATELLITE:
            # Resolve optional status string → SatelliteStatus enum
            status_val = None
            if entry.status:
                try:
                    from app.models.sim_objects import SatelliteStatus
                    status_val = SatelliteStatus(entry.status)
                except ValueError:
                    pass

            ok = fast_update_satellite(
                oid,
                x=sv.x, y=sv.y, z=sv.z,
                vx=sv.vx, vy=sv.vy, vz=sv.vz,
                fuel_kg=entry.fuel_kg,
                status=status_val,
            )

        else:  # DEBRIS
            ok = fast_update_debris(
                oid,
                x=sv.x, y=sv.y, z=sv.z,
                vx=sv.vx, vy=sv.vy, vz=sv.vz,
            )

        if ok:
            processed += 1
            updated.append((oid, entry.object_type.value))
        else:
            unknown_ids.append(oid)

    # ── CDM screening (partial – only pairs involving updated objects) ────────
    active_warnings = screen_updated_objects(updated)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    if unknown_ids:
        logger.debug(
            "Telemetry frame: %d processed, %d unknown IDs: %s",
            processed, len(unknown_ids), unknown_ids,
        )

    return TelemetryACK(
        status              = "ACK",
        processed_count     = processed,
        active_cdm_warnings = active_warnings,
        frame_timestamp     = frame.timestamp.isoformat(),
        processing_time_ms  = round(elapsed_ms, 3),
        unknown_ids         = unknown_ids,
    )


# ═══════════════════════════════════════════════════════════════════════════
# CDM warning query
# ═══════════════════════════════════════════════════════════════════════════

@router.get(
    "/cdm/warnings",
    summary="List active CDM (Conjunction Data Message) warnings",
)
async def get_cdm_warnings():
    """
    Return all currently active conjunction warnings, sorted by miss distance.
    Warnings expire automatically after 5 minutes without a refresh.
    """
    from app.services.cdm_service import active_warnings
    warnings = active_warnings()
    return {
        "active_cdm_warnings": len(warnings),
        "warnings": [w.to_dict() for w in warnings],
    }


@router.get(
    "/cdm/count",
    summary="Quick CDM warning count",
)
async def get_cdm_count():
    """Lightweight endpoint – returns only the active warning count (O(1))."""
    return {"active_cdm_warnings": active_warning_count()}


# ═══════════════════════════════════════════════════════════════════════════
# Historical / subsystem telemetry (secondary endpoints)
# ═══════════════════════════════════════════════════════════════════════════

@router.post(
    "/record",
    response_model=TelemetryRecord,
    status_code=201,
    summary="Ingest a single subsystem telemetry record (historical buffer)",
)
async def ingest_record(record: TelemetryRecord):
    """
    Store one subsystem telemetry record (power, thermal, comms, etc.)
    in the per-satellite ring buffer.  Separate from the high-frequency
    position/velocity ingest above.
    """
    return _svc.ingest(record)


@router.post(
    "/record/batch",
    summary="Ingest a batch of subsystem telemetry records",
)
async def ingest_batch(batch: TelemetryBatch):
    records = _svc.ingest_batch(batch)
    return {"ingested": len(records), "satellite_id": batch.satellite_id}


@router.get(
    "/{satellite_id}",
    response_model=List[TelemetryRecord],
    summary="Query subsystem telemetry records for a satellite",
)
async def query_telemetry(
    satellite_id: str,
    start: Optional[datetime] = Query(None, description="ISO8601 start timestamp"),
    end:   Optional[datetime] = Query(None, description="ISO8601 end timestamp"),
    limit: int                = Query(500, ge=1, le=5000),
):
    return _svc.query(satellite_id, start=start, end=end, limit=limit)


@router.get(
    "/{satellite_id}/latest",
    response_model=TelemetryRecord,
    summary="Latest subsystem telemetry record for a satellite",
)
async def latest(satellite_id: str):
    rec = _svc.latest(satellite_id)
    if not rec:
        raise HTTPException(404, f"No telemetry found for satellite '{satellite_id}'")
    return rec


@router.get(
    "/{satellite_id}/stats",
    summary="Subsystem telemetry statistics for a satellite",
)
async def stats_by_satellite(satellite_id: str):
    return _svc.statistics(satellite_id)
