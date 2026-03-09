"""
Satellite Service – CRUD and orbital propagation for constellation satellites.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, List, Optional

from app.models.satellite import Satellite, SatelliteCreate, SatelliteUpdate, SatelliteStatus
from app.models.orbital_state import OrbitalState
from app.physics.propagator import OrbitalPropagator

logger = logging.getLogger(__name__)

# In-memory store (replace with DB adapter in production)
_store: Dict[str, Satellite] = {}


class SatelliteService:
    def __init__(self):
        self._propagator = OrbitalPropagator()

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def create(self, payload: SatelliteCreate) -> Satellite:
        sat = Satellite(**payload.model_dump())
        _store[sat.id] = sat
        logger.info(f"Created satellite {sat.name} [{sat.id}]")
        return sat

    def get(self, satellite_id: str) -> Optional[Satellite]:
        return _store.get(satellite_id)

    def list_all(
        self,
        constellation: Optional[str] = None,
        status: Optional[SatelliteStatus] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Satellite]:
        sats = list(_store.values())
        if constellation:
            sats = [s for s in sats if s.constellation == constellation]
        if status:
            sats = [s for s in sats if s.status == status]
        return sats[offset: offset + limit]

    def update(self, satellite_id: str, payload: SatelliteUpdate) -> Optional[Satellite]:
        sat = _store.get(satellite_id)
        if not sat:
            return None
        updated_data = sat.model_dump()
        updated_data.update({k: v for k, v in payload.model_dump().items() if v is not None})
        updated_data["updated_at"] = datetime.utcnow()
        _store[satellite_id] = Satellite(**updated_data)
        return _store[satellite_id]

    def delete(self, satellite_id: str) -> bool:
        if satellite_id in _store:
            del _store[satellite_id]
            return True
        return False

    def count(self) -> int:
        return len(_store)

    # ------------------------------------------------------------------
    # Orbital operations
    # ------------------------------------------------------------------

    def propagate(
        self,
        satellite_id: str,
        duration_seconds: float,
    ) -> List[OrbitalState]:
        sat = _store.get(satellite_id)
        if not sat:
            raise KeyError(f"Satellite {satellite_id} not found")
        return self._propagator.propagate(
            sat.orbital_elements,
            epoch=sat.updated_at,
            duration_seconds=duration_seconds,
            drag_coefficient=sat.drag_coefficient,
            mass_kg=sat.mass_kg,
            cross_section_m2=sat.cross_section_m2,
        )

    def current_state(self, satellite_id: str) -> OrbitalState:
        sat = _store.get(satellite_id)
        if not sat:
            raise KeyError(f"Satellite {satellite_id} not found")
        from app.physics.coordinate_transforms import keplerian_to_cartesian
        cart = keplerian_to_cartesian(sat.orbital_elements)
        import math
        speed = math.sqrt(cart.vx_km_s**2 + cart.vy_km_s**2 + cart.vz_km_s**2)
        alt = sat.orbital_elements.semi_major_axis_km - 6371.0
        return OrbitalState(
            epoch=datetime.utcnow(),
            keplerian=sat.orbital_elements,
            cartesian=cart,
            altitude_km=alt,
            speed_km_s=speed,
        )

    def get_all_for_store(self) -> Dict[str, Satellite]:
        return _store
