"""
Maneuver planning domain models.
"""

from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field
import uuid


class ManeuverType(str, Enum):
    HOHMANN = "hohmann"
    PLANE_CHANGE = "plane_change"
    STATION_KEEPING = "station_keeping"
    COLLISION_AVOIDANCE = "collision_avoidance"
    DEORBIT = "deorbit"
    PHASING = "phasing"
    CUSTOM = "custom"


class ManeuverStatus(str, Enum):
    PLANNED = "planned"
    APPROVED = "approved"
    UPLOADING = "uploading"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class BurnSegment(BaseModel):
    """A single thruster burn within a multi-burn maneuver sequence."""
    burn_index: int = Field(..., ge=0)
    epoch: datetime
    delta_v_x_m_s: float = 0.0
    delta_v_y_m_s: float = 0.0
    delta_v_z_m_s: float = 0.0
    duration_seconds: Optional[float] = None

    @property
    def delta_v_magnitude_m_s(self) -> float:
        return (self.delta_v_x_m_s ** 2 + self.delta_v_y_m_s ** 2 + self.delta_v_z_m_s ** 2) ** 0.5


class ManeuverCreate(BaseModel):
    satellite_id: str
    maneuver_type: ManeuverType
    target_altitude_km: Optional[float] = Field(None, gt=0)
    target_inclination_deg: Optional[float] = Field(None, ge=0, le=180)
    target_raan_deg: Optional[float] = Field(None, ge=0, lt=360)
    execution_epoch: Optional[datetime] = None
    burns: List[BurnSegment] = Field(default_factory=list)
    notes: Optional[str] = None


class ManeuverPlan(ManeuverCreate):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    status: ManeuverStatus = ManeuverStatus.PLANNED
    total_delta_v_m_s: float = 0.0
    propellant_cost_kg: float = 0.0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = {"populate_by_name": True}
