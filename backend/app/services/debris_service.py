"""
Debris Service – CRUD and risk classification for space debris objects.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from app.models.debris import DebrisObject, DebrisCreate, DebrisRiskLevel
from app.config import settings

logger = logging.getLogger(__name__)

_store: Dict[str, DebrisObject] = {}


class DebrisService:
    def create(self, payload: DebrisCreate) -> DebrisObject:
        obj = DebrisObject(**payload.model_dump())
        obj = self._classify_risk(obj)
        _store[obj.id] = obj
        logger.info(f"Tracked debris {obj.designation} [{obj.id}] risk={obj.risk_level}")
        return obj

    def get(self, debris_id: str) -> Optional[DebrisObject]:
        return _store.get(debris_id)

    def list_all(
        self,
        risk_level: Optional[DebrisRiskLevel] = None,
        tracked_only: bool = True,
        limit: int = 200,
        offset: int = 0,
    ) -> List[DebrisObject]:
        items = list(_store.values())
        if tracked_only:
            items = [d for d in items if d.tracked]
        if risk_level:
            items = [d for d in items if d.risk_level == risk_level]
        return items[offset: offset + limit]

    def delete(self, debris_id: str) -> bool:
        if debris_id in _store:
            del _store[debris_id]
            return True
        return False

    def count(self) -> int:
        return len(_store)

    def get_all_for_store(self) -> Dict[str, DebrisObject]:
        return _store

    # ------------------------------------------------------------------
    # Risk classification
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_risk(obj: DebrisObject) -> DebrisObject:
        alt = obj.orbital_elements.semi_major_axis_km - 6371.0
        size = obj.size_m or 0.1

        if alt < 400 or size >= 1.0:
            obj.risk_level = DebrisRiskLevel.CRITICAL
        elif alt < 600 or size >= 0.1:
            obj.risk_level = DebrisRiskLevel.HIGH
        elif alt < 900:
            obj.risk_level = DebrisRiskLevel.MEDIUM
        else:
            obj.risk_level = DebrisRiskLevel.LOW
        return obj
