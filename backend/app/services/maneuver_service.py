"""
Maneuver Service – plan, validate, approve and execute orbital maneuvers.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, List, Optional

from app.config import settings
from app.models.maneuver import ManeuverPlan, ManeuverCreate, ManeuverStatus, ManeuverType
from app.physics.maneuver_calculator import ManeuverCalculator

logger = logging.getLogger(__name__)

_store: Dict[str, ManeuverPlan] = {}
_calc = ManeuverCalculator()


class ManeuverService:
    def plan(
        self,
        payload: ManeuverCreate,
        wet_mass_kg: float = 100.0,
        isp_s: float = 220.0,
    ) -> ManeuverPlan:
        plan = ManeuverPlan(**payload.model_dump())
        # Auto-compute Δv for Hohmann transfers
        if plan.maneuver_type == ManeuverType.HOHMANN and plan.target_altitude_km:
            from app.services.satellite_service import SatelliteService
            sat_svc = SatelliteService()
            sat = sat_svc.get(plan.satellite_id)
            if sat:
                result = _calc.build_hohmann_plan(
                    satellite_id=plan.satellite_id,
                    current_elements=sat.orbital_elements,
                    target_altitude_km=plan.target_altitude_km,
                    wet_mass_kg=sat.mass_kg,
                    isp_s=sat.isp_seconds,
                )
                plan.total_delta_v_m_s = result["total_delta_v_m_s"]
                plan.propellant_cost_kg = result["propellant_cost_kg"]

        self._validate(plan)
        _store[plan.id] = plan
        logger.info(f"Maneuver {plan.id} planned for satellite {plan.satellite_id}")
        return plan

    def get(self, maneuver_id: str) -> Optional[ManeuverPlan]:
        return _store.get(maneuver_id)

    def list_all(
        self,
        satellite_id: Optional[str] = None,
        status: Optional[ManeuverStatus] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[ManeuverPlan]:
        items = list(_store.values())
        if satellite_id:
            items = [m for m in items if m.satellite_id == satellite_id]
        if status:
            items = [m for m in items if m.status == status]
        return items[offset: offset + limit]

    def approve(self, maneuver_id: str) -> Optional[ManeuverPlan]:
        return self._transition(maneuver_id, ManeuverStatus.APPROVED)

    def cancel(self, maneuver_id: str) -> Optional[ManeuverPlan]:
        return self._transition(maneuver_id, ManeuverStatus.CANCELLED)

    def mark_completed(self, maneuver_id: str) -> Optional[ManeuverPlan]:
        return self._transition(maneuver_id, ManeuverStatus.COMPLETED)

    def mark_failed(self, maneuver_id: str) -> Optional[ManeuverPlan]:
        return self._transition(maneuver_id, ManeuverStatus.FAILED)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _transition(self, maneuver_id: str, new_status: ManeuverStatus) -> Optional[ManeuverPlan]:
        plan = _store.get(maneuver_id)
        if not plan:
            return None
        plan.status = new_status
        plan.updated_at = datetime.utcnow()
        _store[maneuver_id] = plan
        return plan

    @staticmethod
    def _validate(plan: ManeuverPlan):
        if plan.total_delta_v_m_s > settings.MAX_DELTA_V_M_S:
            raise ValueError(
                f"Requested Δv {plan.total_delta_v_m_s:.1f} m/s exceeds limit "
                f"{settings.MAX_DELTA_V_M_S} m/s"
            )
