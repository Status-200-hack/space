"""
Satellite domain model.
"""

from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field
import uuid

from app.models.orbital_state import KeplerianElements


class SatelliteStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    MANOEUVRING = "manoeuvring"
    SAFE_MODE = "safe_mode"
    DECOMMISSIONED = "decommissioned"


class SatelliteBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=128, description="Unique satellite name")
    norad_id: Optional[int] = Field(None, ge=1, description="NORAD catalogue number")
    constellation: Optional[str] = Field(None, description="Constellation family name")
    mass_kg: float = Field(..., gt=0, description="Satellite wet mass [kg]")
    cross_section_m2: float = Field(..., gt=0, description="Mean cross-sectional area [m²]")
    drag_coefficient: float = Field(2.2, gt=0, description="Drag coefficient C_D")
    propellant_kg: float = Field(0.0, ge=0, description="Remaining propellant [kg]")
    isp_seconds: float = Field(220.0, gt=0, description="Specific impulse [s]")
    tle_line1: Optional[str] = Field(None, max_length=69)
    tle_line2: Optional[str] = Field(None, max_length=69)
    tags: List[str] = Field(default_factory=list)


class SatelliteCreate(SatelliteBase):
    orbital_elements: KeplerianElements


class SatelliteUpdate(BaseModel):
    name: Optional[str] = None
    status: Optional[SatelliteStatus] = None
    mass_kg: Optional[float] = None
    propellant_kg: Optional[float] = None
    tle_line1: Optional[str] = None
    tle_line2: Optional[str] = None
    tags: Optional[List[str]] = None


class Satellite(SatelliteBase):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    status: SatelliteStatus = SatelliteStatus.ACTIVE
    orbital_elements: KeplerianElements
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = {"populate_by_name": True}
