"""
Orbital State models – Keplerian elements and Cartesian state vectors.
"""

from __future__ import annotations
from datetime import datetime
from pydantic import BaseModel, Field


class KeplerianElements(BaseModel):
    """Classical Keplerian orbital elements."""

    semi_major_axis_km: float = Field(..., gt=0, description="Semi-major axis [km]")
    eccentricity: float = Field(..., ge=0, lt=1, description="Orbital eccentricity (0=circular)")
    inclination_deg: float = Field(..., ge=0, le=180, description="Inclination [°]")
    raan_deg: float = Field(..., ge=0, lt=360, description="Right Ascension of Ascending Node [°]")
    arg_of_perigee_deg: float = Field(..., ge=0, lt=360, description="Argument of perigee [°]")
    true_anomaly_deg: float = Field(..., ge=0, lt=360, description="True anomaly [°]")

    @property
    def altitude_km(self) -> float:
        from app.config import settings
        return self.semi_major_axis_km - settings.EARTH_RADIUS_KM

    @property
    def period_seconds(self) -> float:
        import math
        from app.config import settings
        return 2 * math.pi * math.sqrt(self.semi_major_axis_km ** 3 / settings.EARTH_MU_KM3_S2)

    model_config = {"populate_by_name": True}


class CartesianState(BaseModel):
    """ECI (Earth-Centred Inertial) position and velocity vectors."""

    x_km: float = Field(..., description="X position [km]")
    y_km: float = Field(..., description="Y position [km]")
    z_km: float = Field(..., description="Z position [km]")
    vx_km_s: float = Field(..., description="X velocity [km/s]")
    vy_km_s: float = Field(..., description="Y velocity [km/s]")
    vz_km_s: float = Field(..., description="Z velocity [km/s]")


class OrbitalState(BaseModel):
    """Full orbital state at a given epoch."""

    epoch: datetime
    keplerian: KeplerianElements
    cartesian: CartesianState | None = None
    altitude_km: float | None = None
    speed_km_s: float | None = None
    reference_frame: str = "ECI"
