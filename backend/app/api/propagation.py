"""
Orbit Propagation API
=====================

Exposes the orbit_propagator engine via REST.

Endpoints
---------
  POST /api/v1/propagation/step
      Single-step propagation for an arbitrary StateVector.

  POST /api/v1/propagation/simulate/{object_id}
      Full-orbit simulation for a registered satellite or debris object.

  GET  /api/v1/propagation/period
      Compute Keplerian orbital period for a given altitude.

  GET  /api/v1/propagation/circular-velocity
      Circular orbit velocity at a given altitude.

  POST /api/v1/propagation/state-from-altitude
      Generate a circular-orbit StateVector for quick testing.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional, Union

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.models.sim_objects import StateVector, SimSatellite, SimDebris, ObjectType
from app.physics.orbit_propagator import (
    propagate_state,
    simulate_orbit,
    orbital_period_s,
    circular_velocity_km_s,
    state_from_altitude,
    PropagationResult,
    OrbitSimulationResult,
)
from app.data.registry import get_satellite, get_debris

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/propagation")


# ═══════════════════════════════════════════════════════════════════════════
# Request / Response schemas
# ═══════════════════════════════════════════════════════════════════════════

class StepRequest(BaseModel):
    """Request body for a single RK4 integration step."""
    state:         StateVector
    dt:            float         = Field(..., gt=0, description="Time step [s]")
    include_j2:    bool          = Field(True,  description="Apply J2 perturbation")
    include_drag:  bool          = Field(True,  description="Apply atmospheric drag")
    drag_coeff:    float         = Field(2.2,   gt=0, description="Drag coefficient C_D")
    area_m2:       float         = Field(2.0,   gt=0, description="Cross-section area [m²]")
    mass_kg:       float         = Field(100.0, gt=0, description="Object mass [kg]")


class StepResponse(BaseModel):
    """Response from a single propagation step."""
    initial_state:  StateVector
    final_state:    StateVector
    dt:             float
    altitude_km:    float
    speed_km_s:     float
    reentry:        bool
    accelerations:  dict


class SimulateRequest(BaseModel):
    """Request body for a full-orbit simulation."""
    time_seconds:  float         = Field(..., gt=0, le=604800,
                                         description="Total simulation duration [s] (max 7 days)")
    step_seconds:  float         = Field(60.0, gt=0, le=3600,
                                         description="RK4 step size [s]")
    object_type:   ObjectType    = Field(ObjectType.SATELLITE,
                                         description="SATELLITE or DEBRIS")
    include_j2:    bool          = Field(True)
    include_drag:  bool          = Field(True)
    drag_coeff:    float         = Field(2.2,  gt=0)
    area_m2:       float         = Field(2.0,  gt=0)
    mass_kg:       Optional[float] = Field(None, gt=0,
                                           description="Override mass [kg]. Auto-detected if None.")
    record_every:  int           = Field(1, ge=1, le=100,
                                         description="Snapshot down-sample factor")


class SimulateSummary(BaseModel):
    """Lightweight summary returned by the simulate endpoint."""
    object_id:        str
    object_type:      str
    epoch_start:      str
    epoch_end:        str
    steps:            int
    trajectory_points: int
    reentry_detected: bool
    reentry_time_s:   Optional[float]
    final_state:      StateVector
    min_altitude_km:  float
    max_altitude_km:  float
    avg_speed_km_s:   float


# ═══════════════════════════════════════════════════════════════════════════
# Endpoint implementations
# ═══════════════════════════════════════════════════════════════════════════

@router.post(
    "/step",
    response_model=StepResponse,
    summary="Single RK4 integration step (arbitrary StateVector)",
)
async def step_propagate(req: StepRequest) -> StepResponse:
    """
    Advance an arbitrary ECI state vector by *dt* seconds using the RK4
    orbit propagator with optional J2 and atmospheric drag.

    No registry lookup is needed – you supply the state directly.
    Useful for testing, visualization pipelines, and custom integrators.
    """
    result = propagate_state(
        req.state,
        req.dt,
        include_j2   = req.include_j2,
        include_drag = req.include_drag,
        drag_coeff   = req.drag_coeff,
        area_m2      = req.area_m2,
        mass_kg      = req.mass_kg,
    )
    return StepResponse(
        initial_state = req.state,
        final_state   = result.state,
        dt            = result.dt,
        altitude_km   = result.altitude_km,
        speed_km_s    = result.speed_km_s,
        reentry       = result.reentry,
        accelerations = result.accelerations,
    )


@router.post(
    "/simulate/{object_id}",
    summary="Simulate full orbit for a registered satellite or debris object",
)
async def simulate_object(object_id: str, req: SimulateRequest) -> SimulateSummary:
    """
    Look up the satellite or debris object in the in-memory registry by ID,
    then propagate its orbit for *time_seconds* using the RK4 engine.

    Returns a compact summary with the final state vector, trajectory length,
    altitude statistics, and re-entry detection.  The full trajectory array
    is available via ``GET /propagation/trajectory/{object_id}`` (see below).
    """
    obj: Union[SimSatellite, SimDebris, None] = None

    if req.object_type == ObjectType.SATELLITE:
        obj = get_satellite(object_id)
        if obj is None:
            raise HTTPException(404, f"Satellite '{object_id}' not found in registry")
    else:
        obj = get_debris(object_id)
        if obj is None:
            raise HTTPException(404, f"Debris '{object_id}' not found in registry")

    try:
        result: OrbitSimulationResult = simulate_orbit(
            obj,
            time_seconds = req.time_seconds,
            step_seconds = req.step_seconds,
            include_j2   = req.include_j2,
            include_drag = req.include_drag,
            drag_coeff   = req.drag_coeff,
            area_m2      = req.area_m2,
            mass_kg      = req.mass_kg,
            record_every = req.record_every,
            epoch        = datetime.utcnow(),
        )
    except Exception as exc:
        logger.exception("simulate_orbit failed for %s: %s", object_id, exc)
        raise HTTPException(500, f"Propagation error: {exc}") from exc

    alts  = result.altitudes_km
    spds  = result.speeds_km_s

    return SimulateSummary(
        object_id         = result.object_id,
        object_type       = result.object_type,
        epoch_start       = result.epoch_start.isoformat(),
        epoch_end         = result.epoch_end.isoformat(),
        steps             = result.steps,
        trajectory_points = len(result.trajectory),
        reentry_detected  = result.reentry_detected,
        reentry_time_s    = result.reentry_time_s,
        final_state       = result.final_state,
        min_altitude_km   = round(min(alts), 3) if alts else 0.0,
        max_altitude_km   = round(max(alts), 3) if alts else 0.0,
        avg_speed_km_s    = round(sum(spds) / len(spds), 4) if spds else 0.0,
    )


@router.post(
    "/simulate/{object_id}/trajectory",
    summary="Simulate orbit and return the full trajectory array",
)
async def simulate_object_trajectory(object_id: str, req: SimulateRequest) -> dict:
    """
    Same as ``/simulate/{object_id}`` but also returns the full trajectory
    as a list of state vectors.  Can produce large responses for long
    simulations – use *record_every* to down-sample.
    """
    obj: Union[SimSatellite, SimDebris, None] = None

    if req.object_type == ObjectType.SATELLITE:
        obj = get_satellite(object_id)
        if obj is None:
            raise HTTPException(404, f"Satellite '{object_id}' not found in registry")
    else:
        obj = get_debris(object_id)
        if obj is None:
            raise HTTPException(404, f"Debris '{object_id}' not found in registry")

    result = simulate_orbit(
        obj,
        time_seconds = req.time_seconds,
        step_seconds = req.step_seconds,
        include_j2   = req.include_j2,
        include_drag = req.include_drag,
        drag_coeff   = req.drag_coeff,
        area_m2      = req.area_m2,
        mass_kg      = req.mass_kg,
        record_every = req.record_every,
        epoch        = datetime.utcnow(),
    )
    return result.to_dict()


# ═══════════════════════════════════════════════════════════════════════════
# Utility / calculator endpoints
# ═══════════════════════════════════════════════════════════════════════════

@router.get(
    "/period",
    summary="Compute orbital period for a given altitude",
)
async def get_orbital_period(
    altitude_km: float = Query(..., gt=0, description="Orbit altitude above Earth [km]"),
):
    """
    Kepler's third law: T = 2π √(a³/μ)  where a = Rₑ + altitude_km.
    """
    a = 6_371.0 + altitude_km
    period_s = orbital_period_s(a)
    return {
        "altitude_km":    altitude_km,
        "semi_major_axis_km": round(a, 3),
        "period_s":       round(period_s, 3),
        "period_min":     round(period_s / 60.0, 3),
        "orbits_per_day": round(86400.0 / period_s, 4),
    }


@router.get(
    "/circular-velocity",
    summary="Circular orbit velocity at a given altitude",
)
async def get_circular_velocity(
    altitude_km: float = Query(..., gt=0, description="Orbit altitude above Earth [km]"),
):
    """v = √(μ / (Rₑ + h))"""
    v = circular_velocity_km_s(altitude_km)
    return {
        "altitude_km": altitude_km,
        "velocity_km_s": round(v, 6),
        "velocity_m_s":  round(v * 1000.0, 3),
    }


@router.post(
    "/state-from-altitude",
    response_model=StateVector,
    summary="Build a circular-orbit StateVector at a given altitude",
)
async def build_state_from_altitude(
    altitude_km:     float = Query(..., gt=0),
    inclination_deg: float = Query(0.0, ge=0, le=180),
):
    """
    Generates an initial ECI StateVector for a circular orbit at *altitude_km*
    with the spacecraft at the +X intercept (0° true anomaly) of the orbit.
    """
    return state_from_altitude(altitude_km, inclination_deg)
