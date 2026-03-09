"""
Models package – Pydantic schemas and domain enumerations.
"""
from app.models.satellite import Satellite, SatelliteCreate, SatelliteUpdate, SatelliteStatus
from app.models.debris import DebrisObject, DebrisCreate, DebrisRiskLevel
from app.models.telemetry import TelemetryRecord, TelemetryBatch
from app.models.maneuver import ManeuverPlan, ManeuverCreate, ManeuverType, ManeuverStatus
from app.models.orbital_state import OrbitalState, KeplerianElements, CartesianState

# ── Simulation-layer Cartesian models ────────────────────────────────────────
from app.models.sim_objects import (
    StateVector,
    SimSatellite,
    SimDebris,
    ObjectType,
    TelemetryObject,
    TelemetryRequest as SimTelemetryRequest,
    SatelliteStatus as SimSatelliteStatus,
)

__all__ = [
    # Keplerian / orbital mechanics models
    "Satellite", "SatelliteCreate", "SatelliteUpdate", "SatelliteStatus",
    "DebrisObject", "DebrisCreate", "DebrisRiskLevel",
    "TelemetryRecord", "TelemetryBatch",
    "ManeuverPlan", "ManeuverCreate", "ManeuverType", "ManeuverStatus",
    "OrbitalState", "KeplerianElements", "CartesianState",

    # Simulation Cartesian models
    "StateVector",
    "SimSatellite",
    "SimDebris",
    "ObjectType",
    "TelemetryObject",
    "SimTelemetryRequest",
    "SimSatelliteStatus",
]
