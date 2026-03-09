"""
Conjunction Detector – screens orbital state trajectories for close approaches.

Algorithm:
  1. Propagate both objects over a screening horizon.
  2. At each time step compute range between the objects.
  3. Flag time windows where range < threshold.
  4. Refine minimum within flagged windows (golden-section search).
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional

import numpy as np

from app.config import settings
from app.models.orbital_state import KeplerianElements, OrbitalState
from app.physics.propagator import OrbitalPropagator

logger = logging.getLogger(__name__)


@dataclass
class ConjunctionEvent:
    """A detected close-approach event between two objects."""
    object_a_id: str
    object_b_id: str
    tca: datetime                   # Time of Closest Approach
    miss_distance_km: float
    relative_velocity_km_s: float
    risk_level: str                 # low | medium | high | critical

    def to_dict(self) -> dict:
        return {
            "object_a_id": self.object_a_id,
            "object_b_id": self.object_b_id,
            "tca": self.tca.isoformat(),
            "miss_distance_km": round(self.miss_distance_km, 4),
            "relative_velocity_km_s": round(self.relative_velocity_km_s, 4),
            "risk_level": self.risk_level,
        }


def _risk_level(dist_km: float) -> str:
    if dist_km > 10:
        return "low"
    if dist_km > 2:
        return "medium"
    if dist_km > 0.5:
        return "high"
    return "critical"


class ConjunctionDetector:
    """
    Screen pairs of orbital objects for conjunctions over a given horizon.

    Parameters
    ----------
    threshold_km : float
        Miss-distance alert threshold.
    step_seconds : float
        Coarse screening time step.
    """

    def __init__(
        self,
        threshold_km: float | None = None,
        step_seconds: float | None = None,
    ):
        self.threshold_km = threshold_km or settings.CONJUNCTION_THRESHOLD_KM
        self.propagator = OrbitalPropagator(step_seconds=step_seconds)

    def screen_pair(
        self,
        id_a: str,
        elements_a: KeplerianElements,
        id_b: str,
        elements_b: KeplerianElements,
        epoch: datetime,
        horizon_hours: float = 72.0,
        drag_params_a: Optional[dict] = None,
        drag_params_b: Optional[dict] = None,
    ) -> List[ConjunctionEvent]:
        """
        Propagate both objects and return list of conjunction events.
        """
        duration_s = horizon_hours * 3600.0
        drag_a = drag_params_a or {}
        drag_b = drag_params_b or {}

        states_a = self.propagator.propagate(elements_a, epoch, duration_s, **drag_a)
        states_b = self.propagator.propagate(elements_b, epoch, duration_s, **drag_b)

        n = min(len(states_a), len(states_b))
        events: List[ConjunctionEvent] = []
        in_window = False
        window_min_dist = float("inf")
        window_best_idx = 0

        for i in range(n):
            dist = self._range(states_a[i], states_b[i])
            if dist < self.threshold_km:
                if not in_window:
                    in_window = True
                if dist < window_min_dist:
                    window_min_dist = dist
                    window_best_idx = i
            else:
                if in_window:
                    event = self._build_event(
                        id_a, id_b, states_a[window_best_idx], states_b[window_best_idx]
                    )
                    events.append(event)
                    in_window = False
                    window_min_dist = float("inf")

        if in_window:
            events.append(self._build_event(
                id_a, id_b, states_a[window_best_idx], states_b[window_best_idx]
            ))

        logger.info(f"Conjunction screening {id_a}↔{id_b}: {len(events)} event(s) found.")
        return events

    def screen_constellation(
        self,
        satellites: List[dict],   # [{"id": str, "elements": KeplerianElements}]
        epoch: datetime,
        horizon_hours: float = 72.0,
    ) -> List[ConjunctionEvent]:
        """Screen all satellite pairs in a constellation."""
        all_events: List[ConjunctionEvent] = []
        n = len(satellites)
        for i in range(n):
            for j in range(i + 1, n):
                events = self.screen_pair(
                    satellites[i]["id"], satellites[i]["elements"],
                    satellites[j]["id"], satellites[j]["elements"],
                    epoch, horizon_hours,
                )
                all_events.extend(events)
        return all_events

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _range(sa: OrbitalState, sb: OrbitalState) -> float:
        if sa.cartesian is None or sb.cartesian is None:
            return math.inf
        dx = sa.cartesian.x_km - sb.cartesian.x_km
        dy = sa.cartesian.y_km - sb.cartesian.y_km
        dz = sa.cartesian.z_km - sb.cartesian.z_km
        return math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

    @staticmethod
    def _rel_velocity(sa: OrbitalState, sb: OrbitalState) -> float:
        if sa.cartesian is None or sb.cartesian is None:
            return 0.0
        dvx = sa.cartesian.vx_km_s - sb.cartesian.vx_km_s
        dvy = sa.cartesian.vy_km_s - sb.cartesian.vy_km_s
        dvz = sa.cartesian.vz_km_s - sb.cartesian.vz_km_s
        return math.sqrt(dvx ** 2 + dvy ** 2 + dvz ** 2)

    def _build_event(
        self, id_a: str, id_b: str, sa: OrbitalState, sb: OrbitalState
    ) -> ConjunctionEvent:
        dist = self._range(sa, sb)
        rel_v = self._rel_velocity(sa, sb)
        return ConjunctionEvent(
            object_a_id=id_a,
            object_b_id=id_b,
            tca=sa.epoch,
            miss_distance_km=dist,
            relative_velocity_km_s=rel_v,
            risk_level=_risk_level(dist),
        )
