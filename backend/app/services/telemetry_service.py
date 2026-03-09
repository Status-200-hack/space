"""
Telemetry Service – ingest, buffer, and query satellite telemetry records.
"""

from __future__ import annotations

import logging
from collections import deque
from datetime import datetime, timedelta
from typing import Deque, Dict, List, Optional

from app.config import settings
from app.models.telemetry import TelemetryRecord, TelemetryBatch

logger = logging.getLogger(__name__)

# Per-satellite ring buffers
_buffers: Dict[str, Deque[TelemetryRecord]] = {}


def _get_buffer(satellite_id: str) -> Deque[TelemetryRecord]:
    if satellite_id not in _buffers:
        _buffers[satellite_id] = deque(maxlen=settings.TELEMETRY_BUFFER_SIZE)
    return _buffers[satellite_id]


class TelemetryService:
    def ingest(self, record: TelemetryRecord) -> TelemetryRecord:
        buf = _get_buffer(record.satellite_id)
        buf.append(record)
        return record

    def ingest_batch(self, batch: TelemetryBatch) -> List[TelemetryRecord]:
        results = []
        for rec in batch.records:
            rec.satellite_id = batch.satellite_id
            results.append(self.ingest(rec))
        logger.info(f"Ingested {len(results)} telemetry records for {batch.satellite_id}")
        return results

    def query(
        self,
        satellite_id: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 500,
    ) -> List[TelemetryRecord]:
        buf = _get_buffer(satellite_id)
        records = list(buf)
        if start:
            records = [r for r in records if r.timestamp >= start]
        if end:
            records = [r for r in records if r.timestamp <= end]
        return records[-limit:]

    def latest(self, satellite_id: str) -> Optional[TelemetryRecord]:
        buf = _get_buffer(satellite_id)
        return buf[-1] if buf else None

    def statistics(self, satellite_id: str) -> dict:
        buf = _get_buffer(satellite_id)
        records = list(buf)
        if not records:
            return {"count": 0}
        altitudes = [r.altitude_km for r in records if r.altitude_km]
        bsoc = [r.battery_soc_pct for r in records if r.battery_soc_pct is not None]
        return {
            "count": len(records),
            "start": records[0].timestamp.isoformat(),
            "end": records[-1].timestamp.isoformat(),
            "avg_altitude_km": sum(altitudes) / len(altitudes) if altitudes else None,
            "min_altitude_km": min(altitudes) if altitudes else None,
            "max_altitude_km": max(altitudes) if altitudes else None,
            "avg_battery_soc_pct": sum(bsoc) / len(bsoc) if bsoc else None,
        }

    def all_satellite_ids(self) -> List[str]:
        return list(_buffers.keys())
