"""
Simulation Service – coordinated multi-object simulation runs.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional

from app.config import settings
from app.models.orbital_state import OrbitalState
from app.physics.propagator import OrbitalPropagator
from app.physics.conjunction_detector import ConjunctionDetector, ConjunctionEvent

logger = logging.getLogger(__name__)


class SimulationRun:
    def __init__(self, run_id: str, satellite_ids: List[str], duration_seconds: float):
        self.run_id = run_id
        self.satellite_ids = satellite_ids
        self.duration_seconds = duration_seconds
        self.status: str = "pending"   # pending | running | completed | failed
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.trajectories: Dict[str, List[OrbitalState]] = {}
        self.conjunctions: List[ConjunctionEvent] = []
        self.error: Optional[str] = None


_runs: Dict[str, SimulationRun] = {}


class SimulationService:
    def __init__(self):
        self._propagator = OrbitalPropagator()
        self._detector = ConjunctionDetector()

    def create_run(
        self,
        satellite_ids: List[str],
        duration_seconds: float,
    ) -> SimulationRun:
        run_id = str(uuid.uuid4())
        run = SimulationRun(run_id, satellite_ids, duration_seconds)
        _runs[run_id] = run
        logger.info(f"Created simulation run {run_id} for {len(satellite_ids)} objects")
        return run

    def execute_run(self, run_id: str) -> SimulationRun:
        """
        Synchronous execution – propagate all objects and screen conjunctions.
        For large constellations, move this to a background task.
        """
        from app.services.satellite_service import SatelliteService
        from app.services.debris_service import DebrisService

        run = _runs.get(run_id)
        if not run:
            raise KeyError(f"Simulation run {run_id} not found")

        run.status = "running"
        run.started_at = datetime.utcnow()

        sat_svc = SatelliteService()
        deb_svc = DebrisService()

        try:
            objects = []  # [{"id": str, "elements": KeplerianElements, ...}]

            for sid in run.satellite_ids:
                sat = sat_svc.get(sid)
                if sat:
                    states = self._propagator.propagate(
                        sat.orbital_elements,
                        epoch=datetime.utcnow(),
                        duration_seconds=run.duration_seconds,
                        drag_coefficient=sat.drag_coefficient,
                        mass_kg=sat.mass_kg,
                        cross_section_m2=sat.cross_section_m2,
                    )
                    run.trajectories[sid] = states
                    objects.append({"id": sid, "elements": sat.orbital_elements})
                else:
                    # Try debris
                    deb = deb_svc.get(sid)
                    if deb:
                        states = self._propagator.propagate(
                            deb.orbital_elements,
                            epoch=datetime.utcnow(),
                            duration_seconds=run.duration_seconds,
                        )
                        run.trajectories[sid] = states
                        objects.append({"id": sid, "elements": deb.orbital_elements})

            # Conjunction screening between all object pairs
            if len(objects) >= 2:
                run.conjunctions = self._detector.screen_constellation(
                    objects,
                    epoch=datetime.utcnow(),
                    horizon_hours=run.duration_seconds / 3600.0,
                )

            run.status = "completed"
            run.completed_at = datetime.utcnow()
            logger.info(f"Simulation run {run_id} completed. {len(run.conjunctions)} conjunctions found.")

        except Exception as exc:
            run.status = "failed"
            run.error = str(exc)
            logger.exception(f"Simulation run {run_id} failed: {exc}")

        return run

    def get_run(self, run_id: str) -> Optional[SimulationRun]:
        return _runs.get(run_id)

    def list_runs(self, limit: int = 50) -> List[SimulationRun]:
        return list(_runs.values())[-limit:]
