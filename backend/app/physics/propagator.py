"""
Orbital Propagator – numerical integration of the equations of motion.

Perturbations modelled:
  • Two-body (point-mass gravity)
  • J2 oblateness
  • Atmospheric drag  (exponential density model)
"""

from __future__ import annotations

import math
import logging
from datetime import datetime, timedelta
from typing import List, Tuple

import numpy as np

from app.config import settings
from app.models.orbital_state import CartesianState, KeplerianElements, OrbitalState
from app.physics.coordinate_transforms import keplerian_to_cartesian, cartesian_to_keplerian

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Atmospheric density – simple exponential scale-height model (USSA76 fitted)
# ---------------------------------------------------------------------------
_ATM_LAYERS: List[Tuple[float, float, float]] = [
    # (base_alt_km, scale_height_km, rho0_kg_m3)
    (0,    8.44,   1.225),
    (100,  5.88,   5.604e-7),
    (200,  7.29,   2.789e-10),
    (300,  8.34,   1.916e-12),
    (400,  7.51,   2.803e-13),
    (500,  8.66,   5.215e-14),
    (600, 11.04,   8.942e-15),
    (700, 16.56,   3.170e-15),
    (800, 27.27,   1.492e-15),
]


def _atmospheric_density(altitude_km: float) -> float:
    """Return atmospheric density [kg/m³] at given altitude."""
    if altitude_km >= 1000:
        return 0.0
    layer = _ATM_LAYERS[0]
    for entry in _ATM_LAYERS:
        if altitude_km >= entry[0]:
            layer = entry
        else:
            break
    base_alt, H, rho0 = layer
    return rho0 * math.exp(-(altitude_km - base_alt) / H)


class OrbitalPropagator:
    """
    Runge-Kutta 4 propagator with J2 and drag perturbations.

    Parameters
    ----------
    step_seconds : float
        Integration time step (default from config).
    """

    MU = settings.EARTH_MU_KM3_S2          # km³/s²
    RE = settings.EARTH_RADIUS_KM           # km
    J2 = settings.J2_COEFFICIENT

    def __init__(self, step_seconds: float | None = None):
        self.dt = step_seconds or settings.SIMULATION_STEP_SECONDS

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def propagate(
        self,
        elements: KeplerianElements,
        epoch: datetime,
        duration_seconds: float,
        drag_coefficient: float = 2.2,
        mass_kg: float = 100.0,
        cross_section_m2: float = 1.0,
    ) -> List[OrbitalState]:
        """
        Propagate orbital state from *epoch* for *duration_seconds*.

        Returns a list of OrbitalState snapshots at every time step.
        """
        r, v = self._kep_to_rv(elements)
        state = np.concatenate([r, v])
        t = 0.0
        results: List[OrbitalState] = []
        B = (drag_coefficient * cross_section_m2) / mass_kg  # ballistic coefficient inverse

        while t <= duration_seconds:
            r_vec = state[:3]
            v_vec = state[3:]
            altitude = np.linalg.norm(r_vec) - self.RE
            speed = np.linalg.norm(v_vec)

            kep = cartesian_to_keplerian(
                CartesianState(
                    x_km=r_vec[0], y_km=r_vec[1], z_km=r_vec[2],
                    vx_km_s=v_vec[0], vy_km_s=v_vec[1], vz_km_s=v_vec[2],
                )
            )
            results.append(OrbitalState(
                epoch=epoch + timedelta(seconds=t),
                keplerian=kep,
                cartesian=CartesianState(
                    x_km=r_vec[0], y_km=r_vec[1], z_km=r_vec[2],
                    vx_km_s=v_vec[0], vy_km_s=v_vec[1], vz_km_s=v_vec[2],
                ),
                altitude_km=float(altitude),
                speed_km_s=float(speed),
            ))

            if altitude < 0:
                logger.warning("Object re-entered atmosphere – propagation stopped.")
                break

            state = self._rk4_step(state, self.dt, B)
            t += self.dt

        return results

    def propagate_to_epoch(
        self,
        elements: KeplerianElements,
        start_epoch: datetime,
        target_epoch: datetime,
        **kwargs,
    ) -> OrbitalState:
        duration = (target_epoch - start_epoch).total_seconds()
        if duration <= 0:
            raise ValueError("target_epoch must be after start_epoch")
        states = self.propagate(elements, start_epoch, duration, **kwargs)
        return states[-1]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _rk4_step(self, state: np.ndarray, dt: float, B: float) -> np.ndarray:
        k1 = self._derivatives(state, B)
        k2 = self._derivatives(state + 0.5 * dt * k1, B)
        k3 = self._derivatives(state + 0.5 * dt * k2, B)
        k4 = self._derivatives(state + dt * k3, B)
        return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def _derivatives(self, state: np.ndarray, B: float) -> np.ndarray:
        r = state[:3]
        v = state[3:]
        r_norm = np.linalg.norm(r)
        r2 = r_norm ** 2

        # Two-body acceleration
        a_2body = -self.MU / r2 * (r / r_norm)

        # J2 perturbation
        z2_r2 = (r[2] / r_norm) ** 2
        j2_factor = 1.5 * self.J2 * self.MU * self.RE ** 2 / (r2 ** 2)
        a_j2 = j2_factor * np.array([
            r[0] / r_norm * (5 * z2_r2 - 1),
            r[1] / r_norm * (5 * z2_r2 - 1),
            r[2] / r_norm * (5 * z2_r2 - 3),
        ])

        # Atmospheric drag (km/m unit consistency → convert density to km)
        alt_km = r_norm - self.RE
        rho = _atmospheric_density(alt_km) * 1e9  # kg/km³
        v_norm = np.linalg.norm(v)
        if v_norm > 0 and alt_km < 1000:
            a_drag = -0.5 * B * rho * v_norm * v
        else:
            a_drag = np.zeros(3)

        a_total = a_2body + a_j2 + a_drag
        return np.concatenate([v, a_total])

    @staticmethod
    def _kep_to_rv(kep: KeplerianElements) -> Tuple[np.ndarray, np.ndarray]:
        cart = keplerian_to_cartesian(kep)
        r = np.array([cart.x_km, cart.y_km, cart.z_km])
        v = np.array([cart.vx_km_s, cart.vy_km_s, cart.vz_km_s])
        return r, v
