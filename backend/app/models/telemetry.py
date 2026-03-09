"""
Telemetry data models.
"""

from __future__ import annotations
from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
import uuid


class TelemetryRecord(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    satellite_id: str = Field(..., description="ID of the source satellite")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Attitude & position
    latitude_deg: Optional[float] = Field(None, ge=-90, le=90)
    longitude_deg: Optional[float] = Field(None, ge=-180, le=180)
    altitude_km: Optional[float] = Field(None, gt=0)
    velocity_km_s: Optional[float] = None

    # Power subsystem
    battery_voltage_v: Optional[float] = None
    battery_soc_pct: Optional[float] = Field(None, ge=0, le=100)
    solar_power_w: Optional[float] = None

    # Thermal
    temperature_c: Optional[float] = None

    # Propulsion
    propellant_kg: Optional[float] = Field(None, ge=0)
    thruster_status: Optional[str] = None

    # Signal
    signal_strength_dbm: Optional[float] = None
    bit_error_rate: Optional[float] = Field(None, ge=0, le=1)

    # Generic extensible payload
    custom: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"populate_by_name": True}


class TelemetryBatch(BaseModel):
    satellite_id: str
    records: List[TelemetryRecord] = Field(..., min_length=1)
    source: str = "ground_station"
