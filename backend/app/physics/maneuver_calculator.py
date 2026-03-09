"""
Maneuver Calculator – computes Δv budgets for orbital transfers.

Supported maneuver types:
  • Hohmann transfer (circular-to-circular)
  • Bi-elliptic transfer
  • Plane change (simple & combined)
  • Station-keeping
  • Collision avoidance (RAD/T-dir)
"""

from __future__ import annotations

import math
import logging
from typing import Tuple

from app.config import settings
from app.models.orbital_state import KeplerianElements
from app.models.maneuver import BurnSegment, ManeuverPlan, ManeuverType

logger = logging.getLogger(__name__)

MU = settings.EARTH_MU_KM3_S2
RE = settings.EARTH_RADIUS_KM


def _circular_velocity(sma_km: float) -> float:
    return math.sqrt(MU / sma_km)


def _vis_viva(r_km: float, sma_km: float) -> float:
    return math.sqrt(MU * (2 / r_km - 1 / sma_km))


class ManeuverCalculator:
    """
    Compute Δv budgets and burn segments.
    """

    @staticmethod
    def hohmann_transfer(
        initial_sma_km: float,
        target_sma_km: float,
    ) -> Tuple[float, float, float]:
        """
        Compute Hohmann transfer Δv₁, Δv₂ and total [km/s].
        Both orbits assumed circular.
        """
        v1 = _circular_velocity(initial_sma_km)
        v3 = _circular_velocity(target_sma_km)
        sma_transfer = (initial_sma_km + target_sma_km) / 2
        v_transfer_perigee = _vis_viva(initial_sma_km, sma_transfer)
        v_transfer_apogee = _vis_viva(target_sma_km, sma_transfer)
        dv1 = abs(v_transfer_perigee - v1)
        dv2 = abs(v3 - v_transfer_apogee)
        return dv1, dv2, dv1 + dv2

    @staticmethod
    def plane_change(v_km_s: float, delta_i_deg: float) -> float:
        """Pure plane change Δv [km/s]."""
        delta_i = math.radians(delta_i_deg)
        return 2 * v_km_s * math.sin(delta_i / 2)

    @staticmethod
    def combined_hohmann_plane_change(
        initial_sma_km: float,
        target_sma_km: float,
        delta_i_deg: float,
    ) -> float:
        """Combined Hohmann + inclination change Δv budget."""
        delta_i = math.radians(delta_i_deg)
        v1 = _circular_velocity(initial_sma_km)
        v2 = _circular_velocity(target_sma_km)
        sma_t = (initial_sma_km + target_sma_km) / 2
        v_ap = _vis_viva(target_sma_km, sma_t)
        dv1 = abs(_vis_viva(initial_sma_km, sma_t) - v1)
        dv2 = math.sqrt(v_ap ** 2 + v2 ** 2 - 2 * v_ap * v2 * math.cos(delta_i))
        return dv1 + dv2

    @staticmethod
    def tsiolkovsky_propellant(
        delta_v_km_s: float,
        wet_mass_kg: float,
        isp_s: float,
    ) -> float:
        """Mass of propellant consumed [kg] for a given Δv (Tsiolkovsky)."""
        g0 = 9.80665e-3  # km/s²
        ve = isp_s * g0   # effective exhaust velocity [km/s]
        mass_ratio = math.exp(delta_v_km_s / ve)
        return wet_mass_kg * (1 - 1 / mass_ratio)

    @staticmethod
    def collision_avoidance_dv(
        relative_velocity_km_s: float,
        miss_distance_km: float,
        desired_miss_distance_km: float = 2.0,
    ) -> float:
        """
        Rough Δv estimate for a radial collision avoidance manoeuvre.
        Uses a simple two-impulse approximation.
        """
        if miss_distance_km >= desired_miss_distance_km:
            return 0.0
        scale = (desired_miss_distance_km - miss_distance_km) / desired_miss_distance_km
        return relative_velocity_km_s * scale * 0.01   # heuristic fraction

    def build_hohmann_plan(
        self,
        satellite_id: str,
        current_elements: KeplerianElements,
        target_altitude_km: float,
        wet_mass_kg: float = 100.0,
        isp_s: float = 220.0,
    ) -> dict:
        """
        Return a ready-to-use dict describing a Hohmann maneuver plan.
        """
        current_sma = current_elements.semi_major_axis_km
        target_sma = RE + target_altitude_km
        dv1, dv2, total_dv = self.hohmann_transfer(current_sma, target_sma)
        prop = self.tsiolkovsky_propellant(total_dv, wet_mass_kg, isp_s)
        logger.info(f"Hohmann plan: Δv={total_dv:.4f} km/s, propellant={prop:.3f} kg")
        return {
            "satellite_id": satellite_id,
            "maneuver_type": ManeuverType.HOHMANN,
            "target_altitude_km": target_altitude_km,
            "delta_v1_km_s": dv1,
            "delta_v2_km_s": dv2,
            "total_delta_v_km_s": total_dv,
            "total_delta_v_m_s": total_dv * 1000,
            "propellant_cost_kg": prop,
        }
