"""
In-memory registries for simulation objects.

These are the single source of truth for the live simulation state.

    satellites  : Dict[str, SimSatellite]   – keyed by satellite ID
    debris      : Dict[str, SimDebris]      – keyed by debris ID

Both dicts are module-level singletons intentionally shared across the
entire process lifetime (no DB dependency for local simulation runs).
Replace or wrap these with a DB-backed store for persistence.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, Iterator, List, Optional, Tuple

from app.models.sim_objects import (
    SimSatellite,
    SimDebris,
    SatelliteStatus,
    StateVector,
    TelemetryRequest,
    ObjectType,
)

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Core registries  (module-level singletons)
# ═══════════════════════════════════════════════════════════════════════════

satellites: Dict[str, SimSatellite] = {}
debris:     Dict[str, SimDebris]    = {}


# ═══════════════════════════════════════════════════════════════════════════
# Satellite registry helpers
# ═══════════════════════════════════════════════════════════════════════════

def add_satellite(sat: SimSatellite) -> SimSatellite:
    """Insert or replace a satellite in the registry."""
    satellites[sat.id] = sat
    logger.debug("Registry: satellite added/updated  id=%s  name=%s", sat.id, sat.name)
    return sat


def get_satellite(satellite_id: str) -> Optional[SimSatellite]:
    """Return satellite by ID, or None if not found."""
    return satellites.get(satellite_id)


def remove_satellite(satellite_id: str) -> bool:
    """Remove a satellite. Returns True if it existed."""
    existed = satellite_id in satellites
    satellites.pop(satellite_id, None)
    return existed


def list_satellites(
    status: Optional[SatelliteStatus] = None,
    limit: int = 500,
    offset: int = 0,
) -> List[SimSatellite]:
    """Return a filtered, paginated view of the satellite registry."""
    items = list(satellites.values())
    if status is not None:
        items = [s for s in items if s.status == status]
    return items[offset: offset + limit]


def update_satellite_state(
    satellite_id: str,
    position: Optional[StateVector] = None,
    velocity: Optional[StateVector] = None,
    fuel_kg: Optional[float] = None,
    status: Optional[SatelliteStatus] = None,
) -> Optional[SimSatellite]:
    """
    Patch individual fields on an existing satellite without replacing it.
    Returns the updated object, or None if the ID was not found.
    """
    sat = satellites.get(satellite_id)
    if sat is None:
        return None
    data = sat.model_dump()
    if position is not None:
        data["position"] = position.model_dump()
    if velocity is not None:
        data["velocity"] = velocity.model_dump()
    if fuel_kg is not None:
        data["fuel_kg"] = fuel_kg
    if status is not None:
        data["status"] = status
    data["updated_at"] = datetime.utcnow()
    updated = SimSatellite(**data)
    satellites[satellite_id] = updated
    return updated


# ═══════════════════════════════════════════════════════════════════════════
# Debris registry helpers
# ═══════════════════════════════════════════════════════════════════════════

def add_debris(obj: SimDebris) -> SimDebris:
    """Insert or replace a debris object in the registry."""
    debris[obj.id] = obj
    logger.debug("Registry: debris added/updated  id=%s  designation=%s", obj.id, obj.designation)
    return obj


def get_debris(debris_id: str) -> Optional[SimDebris]:
    """Return debris object by ID, or None if not found."""
    return debris.get(debris_id)


def remove_debris(debris_id: str) -> bool:
    """Remove a debris object. Returns True if it existed."""
    existed = debris_id in debris
    debris.pop(debris_id, None)
    return existed


def list_debris(limit: int = 500, offset: int = 0) -> List[SimDebris]:
    """Return a paginated view of all tracked debris objects."""
    return list(debris.values())[offset: offset + limit]


def update_debris_state(
    debris_id: str,
    position: Optional[StateVector] = None,
    velocity: Optional[StateVector] = None,
) -> Optional[SimDebris]:
    """Patch position / velocity on an existing debris object."""
    obj = debris.get(debris_id)
    if obj is None:
        return None
    data = obj.model_dump()
    if position is not None:
        data["position"] = position.model_dump()
    if velocity is not None:
        data["velocity"] = velocity.model_dump()
    updated = SimDebris(**data)
    debris[debris_id] = updated
    return updated


# ═══════════════════════════════════════════════════════════════════════════
# High-frequency fast-patch helpers  (no model_dump / reconstruct)
# ═══════════════════════════════════════════════════════════════════════════

def fast_update_satellite(
    satellite_id: str,
    x: float, y: float, z: float,
    vx: float, vy: float, vz: float,
    fuel_kg: Optional[float] = None,
    status: Optional[SatelliteStatus] = None,
) -> bool:
    """
    High-throughput in-place state patch for a satellite.

    Mutates position/velocity components directly via object.__setattr__
    (bypassing Pydantic validation) so we avoid the model_dump/reconstruct
    cost on the hot telemetry ingest path.

    Returns True if the satellite was found, False otherwise.
    """
    sat = satellites.get(satellite_id)
    if sat is None:
        return False
    pos = sat.position
    vel = sat.velocity
    object.__setattr__(pos, "x",  x)
    object.__setattr__(pos, "y",  y)
    object.__setattr__(pos, "z",  z)
    object.__setattr__(vel, "vx", vx)
    object.__setattr__(vel, "vy", vy)
    object.__setattr__(vel, "vz", vz)
    if fuel_kg is not None:
        object.__setattr__(sat, "fuel_kg", fuel_kg)
    if status is not None:
        object.__setattr__(sat, "status", status)
    object.__setattr__(sat, "updated_at", datetime.utcnow())
    return True


def fast_update_debris(
    debris_id: str,
    x: float, y: float, z: float,
    vx: float, vy: float, vz: float,
) -> bool:
    """
    High-throughput in-place state patch for a debris object.
    Returns True if the object was found, False otherwise.
    """
    obj = debris.get(debris_id)
    if obj is None:
        return False
    pos = obj.position
    vel = obj.velocity
    object.__setattr__(pos, "x",  x)
    object.__setattr__(pos, "y",  y)
    object.__setattr__(pos, "z",  z)
    object.__setattr__(vel, "vx", vx)
    object.__setattr__(vel, "vy", vy)
    object.__setattr__(vel, "vz", vz)
    return True


# ═══════════════════════════════════════════════════════════════════════════
# Cross-registry helpers
# ═══════════════════════════════════════════════════════════════════════════

def all_objects() -> Iterator[Tuple[str, str, StateVector]]:
    """
    Yield (id, type_label, state_vector) for every tracked object.
    Useful for conjunction screening that needs to iterate all objects.
    """
    for sat in satellites.values():
        sv = StateVector(
            x=sat.position.x, y=sat.position.y, z=sat.position.z,
            vx=sat.velocity.vx, vy=sat.velocity.vy, vz=sat.velocity.vz,
        )
        yield sat.id, "SATELLITE", sv
    for deb in debris.values():
        sv = StateVector(
            x=deb.position.x, y=deb.position.y, z=deb.position.z,
            vx=deb.velocity.vx, vy=deb.velocity.vy, vz=deb.velocity.vz,
        )
        yield deb.id, "DEBRIS", sv


def registry_stats() -> dict:
    """Return a summary of current registry sizes."""
    return {
        "satellite_count": len(satellites),
        "debris_count": len(debris),
        "total_objects": len(satellites) + len(debris),
    }


def clear_all() -> None:
    """Wipe both registries and CDM warnings (used in tests / simulation resets)."""
    from app.services.cdm_service import clear_warnings
    satellites.clear()
    debris.clear()
    clear_warnings()
    logger.warning("Registry: all objects cleared.")


# ═══════════════════════════════════════════════════════════════════════════
# Telemetry ingest
# ═══════════════════════════════════════════════════════════════════════════

def apply_telemetry(request: TelemetryRequest) -> dict:
    """
    Apply a TelemetryRequest frame to the in-memory registries.

    For each entry:
      • SATELLITE → update position, velocity, fuel_kg, status (if present)
      • DEBRIS    → update position and velocity

    Returns a summary dict with counts of updated and unknown objects.
    """
    updated_satellites = 0
    updated_debris = 0
    unknown_ids: List[str] = []

    for entry in request.objects:
        oid = entry.object_id

        if entry.object_type == ObjectType.SATELLITE:
            pos = StateVector(
                x=entry.state.x, y=entry.state.y, z=entry.state.z,
                vx=0.0, vy=0.0, vz=0.0,
            )
            vel = StateVector(
                x=0.0, y=0.0, z=0.0,
                vx=entry.state.vx, vy=entry.state.vy, vz=entry.state.vz,
            )
            status_val = None
            if entry.status:
                try:
                    status_val = SatelliteStatus(entry.status)
                except ValueError:
                    pass

            result = update_satellite_state(
                oid,
                position=pos,
                velocity=vel,
                fuel_kg=entry.fuel_kg,
                status=status_val,
            )
            if result:
                updated_satellites += 1
            else:
                unknown_ids.append(oid)

        else:  # DEBRIS
            pos = StateVector(
                x=entry.state.x, y=entry.state.y, z=entry.state.z,
                vx=0.0, vy=0.0, vz=0.0,
            )
            vel = StateVector(
                x=0.0, y=0.0, z=0.0,
                vx=entry.state.vx, vy=entry.state.vy, vz=entry.state.vz,
            )
            result = update_debris_state(oid, position=pos, velocity=vel)
            if result:
                updated_debris += 1
            else:
                unknown_ids.append(oid)

    logger.info(
        "Telemetry frame at %s: %d satellites, %d debris updated; %d unknown IDs",
        request.timestamp.isoformat(),
        updated_satellites,
        updated_debris,
        len(unknown_ids),
    )

    return {
        "frame_timestamp": request.timestamp.isoformat(),
        "updated_satellites": updated_satellites,
        "updated_debris": updated_debris,
        "unknown_ids": unknown_ids,
    }
