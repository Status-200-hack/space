"""
Core simulation data models.

These Cartesian-state models are the primary representation used by the
real-time orbital simulator.  They sit alongside (not replacing) the
Keplerian-element models used for long-arc propagation.

Hierarchy
---------
StateVector         – raw 6-DOF position + velocity in ECI [km, km/s]
SimSatellite        – satellite carrying a StateVector + fuel budget
SimDebris           – uncontrolled debris object carrying a StateVector
ObjectType          – discriminator enum  (SATELLITE | DEBRIS)
TelemetryObject     – one snapshot entry in a telemetry frame
TelemetryRequest    – full ingest frame  (timestamp + list[TelemetryObject])
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Annotated, List, Union
from pydantic import BaseModel, Field
import uuid


# ═══════════════════════════════════════════════════════════════════════════
# 1.  StateVector
# ═══════════════════════════════════════════════════════════════════════════

class StateVector(BaseModel):
    """
    Earth-Centred Inertial (ECI) 6-DOF state vector.

    Position components are in kilometres; velocity in km/s.
    """

    x: float = Field(..., description="X position [km]")
    y: float = Field(..., description="Y position [km]")
    z: float = Field(..., description="Z position [km]")
    vx: float = Field(..., description="X velocity [km/s]")
    vy: float = Field(..., description="Y velocity [km/s]")
    vz: float = Field(..., description="Z velocity [km/s]")

    # ── Derived helpers (no stored fields) ──────────────────────────────────
    @property
    def position_km(self) -> tuple[float, float, float]:
        """Position tuple (x, y, z) in km."""
        return (self.x, self.y, self.z)

    @property
    def velocity_km_s(self) -> tuple[float, float, float]:
        """Velocity tuple (vx, vy, vz) in km/s."""
        return (self.vx, self.vy, self.vz)

    @property
    def radius_km(self) -> float:
        """Magnitude of position vector [km]."""
        return (self.x**2 + self.y**2 + self.z**2) ** 0.5

    @property
    def speed_km_s(self) -> float:
        """Magnitude of velocity vector [km/s]."""
        return (self.vx**2 + self.vy**2 + self.vz**2) ** 0.5

    @property
    def altitude_km(self) -> float:
        """Approximate altitude above Earth's surface [km]."""
        EARTH_RADIUS_KM = 6_371.0
        return self.radius_km - EARTH_RADIUS_KM

    model_config = {"populate_by_name": True}


# ═══════════════════════════════════════════════════════════════════════════
# 2.  SimSatellite
# ═══════════════════════════════════════════════════════════════════════════

class SatelliteStatus(str, Enum):
    ACTIVE        = "active"
    INACTIVE      = "inactive"
    MANOEUVRING   = "manoeuvring"
    SAFE_MODE     = "safe_mode"
    DECOMMISSIONED = "decommissioned"


class SimSatellite(BaseModel):
    """
    Simulation-layer representation of a satellite.

    Stores Cartesian state + fuel level + operational status.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique satellite identifier (UUID)",
    )
    name: str = Field(..., min_length=1, max_length=128, description="Satellite name")

    # Position [km] and velocity [km/s] as a nested StateVector
    position: StateVector = Field(..., description="ECI position vector [km]")
    velocity: StateVector = Field(
        ...,
        description=(
            "ECI velocity vector [km/s]. "
            "Only vx/vy/vz are meaningful; x/y/z are set to 0.0."
        ),
    )

    fuel_kg: float = Field(..., ge=0.0, description="Remaining propellant mass [kg]")
    status: SatelliteStatus = Field(SatelliteStatus.ACTIVE, description="Operational status")

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # ── Convenience constructor ──────────────────────────────────────────────
    @classmethod
    def from_components(
        cls,
        name: str,
        x: float, y: float, z: float,
        vx: float, vy: float, vz: float,
        fuel_kg: float,
        status: SatelliteStatus = SatelliteStatus.ACTIVE,
        **kwargs,
    ) -> "SimSatellite":
        """Create a SimSatellite from flat position/velocity components."""
        position = StateVector(x=x, y=y, z=z, vx=0.0, vy=0.0, vz=0.0)
        velocity = StateVector(x=0.0, y=0.0, z=0.0, vx=vx, vy=vy, vz=vz)
        return cls(
            name=name,
            position=position,
            velocity=velocity,
            fuel_kg=fuel_kg,
            status=status,
            **kwargs,
        )

    model_config = {"populate_by_name": True}


# ═══════════════════════════════════════════════════════════════════════════
# 3.  SimDebris
# ═══════════════════════════════════════════════════════════════════════════

class SimDebris(BaseModel):
    """
    Simulation-layer representation of an uncontrolled debris object.

    Contains only the state vector: no fuel, no control authority.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique debris identifier (UUID)",
    )
    designation: str = Field(
        ..., description="COSPAR ID, NORAD catalogue number, or informal label"
    )

    # Position [km] and velocity [km/s]
    position: StateVector = Field(..., description="ECI position vector [km]")
    velocity: StateVector = Field(..., description="ECI velocity vector [km/s]")

    created_at: datetime = Field(default_factory=datetime.utcnow)

    # ── Convenience constructor ──────────────────────────────────────────────
    @classmethod
    def from_components(
        cls,
        designation: str,
        x: float, y: float, z: float,
        vx: float, vy: float, vz: float,
        **kwargs,
    ) -> "SimDebris":
        """Create a SimDebris from flat position/velocity components."""
        position = StateVector(x=x, y=y, z=z, vx=0.0, vy=0.0, vz=0.0)
        velocity = StateVector(x=0.0, y=0.0, z=0.0, vx=vx, vy=vy, vz=vz)
        return cls(designation=designation, position=position, velocity=velocity, **kwargs)

    model_config = {"populate_by_name": True}


# ═══════════════════════════════════════════════════════════════════════════
# 4.  Telemetry ingestion models
# ═══════════════════════════════════════════════════════════════════════════

class ObjectType(str, Enum):
    """Discriminator for objects carried inside a TelemetryRequest."""
    SATELLITE = "SATELLITE"
    DEBRIS    = "DEBRIS"


class TelemetryObject(BaseModel):
    """
    A single state snapshot for one tracked object.

    The *object_type* field acts as a discriminator so the backend can
    route the payload to the correct registry.
    """

    object_id: str = Field(..., description="ID of the satellite or debris object")
    object_type: ObjectType = Field(..., description="SATELLITE or DEBRIS")
    state: StateVector = Field(..., description="Current ECI state vector")

    # Optional subsystem readings (null-safe – may be omitted by sensor)
    fuel_kg: float | None = Field(None, ge=0.0, description="Remaining fuel [kg] (satellite only)")
    status: str | None = Field(None, description="Object status string (satellite only)")


class TelemetryRequest(BaseModel):
    """
    A telemetry ingest frame containing one or more object state snapshots.

    Sent by ground stations or simulation engines at the end of each
    propagation step.
    """

    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="ISO 8601 UTC epoch of the telemetry frame",
    )
    objects: List[TelemetryObject] = Field(
        ...,
        min_length=1,
        description="List of object state snapshots (SATELLITE and/or DEBRIS)",
    )
    source: str = Field("simulator", description="Data source identifier")

    # ── Partitioned views (no extra storage) ───────────────────────────────
    @property
    def satellites(self) -> List[TelemetryObject]:
        """Filter to only SATELLITE entries."""
        return [o for o in self.objects if o.object_type == ObjectType.SATELLITE]

    @property
    def debris_objects(self) -> List[TelemetryObject]:
        """Filter to only DEBRIS entries."""
        return [o for o in self.objects if o.object_type == ObjectType.DEBRIS]

    model_config = {"populate_by_name": True}
