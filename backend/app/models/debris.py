"""
Debris object domain model.
"""

from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field
import uuid

from app.models.orbital_state import KeplerianElements


class DebrisRiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DebrisBase(BaseModel):
    designation: str = Field(..., description="COSPAR or informal designation")
    norad_id: Optional[int] = Field(None, ge=1)
    origin: Optional[str] = Field(None, description="Source event / parent object")
    size_m: Optional[float] = Field(None, gt=0, description="Estimated largest dimension [m]")
    mass_kg: Optional[float] = Field(None, gt=0, description="Estimated mass [kg]")
    radar_cross_section_m2: Optional[float] = Field(None, gt=0)
    tle_line1: Optional[str] = Field(None, max_length=69)
    tle_line2: Optional[str] = Field(None, max_length=69)


class DebrisCreate(DebrisBase):
    orbital_elements: KeplerianElements


class DebrisObject(DebrisBase):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    orbital_elements: KeplerianElements
    risk_level: DebrisRiskLevel = DebrisRiskLevel.LOW
    tracked: bool = True
    last_observation: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = {"populate_by_name": True}
