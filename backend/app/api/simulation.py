"""
Simulation API – create/execute simulation runs and step the universe forward.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from typing import List, Optional
from pydantic import BaseModel, Field

from app.services.simulation_service import SimulationService, SimulationRun
from app.services.sim_step_service import get_step_service, reset_step_service, COLLISION_THRESHOLD_KM

router = APIRouter(prefix="/simulation")
_svc = SimulationService()


class SimulationRequest(BaseModel):
    satellite_ids: List[str] = Field(..., min_length=1)
    duration_seconds: float = Field(3600.0, gt=0, le=604800)
    run_async: bool = Field(False, description="Run in background task")


def _run_summary(run: SimulationRun) -> dict:
    return {
        "run_id": run.run_id,
        "satellite_ids": run.satellite_ids,
        "duration_seconds": run.duration_seconds,
        "status": run.status,
        "started_at": run.started_at.isoformat() if run.started_at else None,
        "completed_at": run.completed_at.isoformat() if run.completed_at else None,
        "trajectory_count": {sid: len(traj) for sid, traj in run.trajectories.items()},
        "conjunction_count": len(run.conjunctions),
        "conjunctions": [c.to_dict() for c in run.conjunctions],
        "error": run.error,
    }


@router.post("/run", summary="Create and execute a simulation run")
async def run_simulation(payload: SimulationRequest, background_tasks: BackgroundTasks):
    run = _svc.create_run(payload.satellite_ids, payload.duration_seconds)
    if payload.run_async:
        background_tasks.add_task(_svc.execute_run, run.run_id)
        return {"run_id": run.run_id, "status": "queued"}
    run = _svc.execute_run(run.run_id)
    return _run_summary(run)


@router.get("/run/{run_id}", summary="Get simulation run status and results")
async def get_run(run_id: str):
    run = _svc.get_run(run_id)
    if not run:
        raise HTTPException(404, f"Simulation run '{run_id}' not found")
    return _run_summary(run)


@router.get("/runs", summary="List recent simulation runs")
async def list_runs(limit: int = 50):
    runs = _svc.list_runs(limit=limit)
    return [_run_summary(r) for r in runs]


@router.get("/run/{run_id}/trajectory/{satellite_id}", summary="Get propagated trajectory for one satellite")
async def get_trajectory(run_id: str, satellite_id: str):
    run = _svc.get_run(run_id)
    if not run:
        raise HTTPException(404, f"Simulation run '{run_id}' not found")
    traj = run.trajectories.get(satellite_id)
    if traj is None:
        raise HTTPException(404, f"Satellite '{satellite_id}' not in run '{run_id}'")
    return traj


# ════════════════════════════════════════════════════════════════════════
# Discrete-time simulation stepping  (POST /simulation/step)
# ════════════════════════════════════════════════════════════════════════

class StepRequest(BaseModel):
    step_seconds: float = Field(
        60.0, gt=0, le=86400,
        description="Simulation step duration [s] (max 86400 / 1 day)",
    )
    include_j2:          bool  = Field(True,  description="Apply J2 oblateness perturbation")
    include_drag:        bool  = Field(True,  description="Apply atmospheric drag")
    collision_threshold_km: float = Field(
        COLLISION_THRESHOLD_KM, gt=0,
        description="Miss-distance threshold for collision events [km]",
    )


@router.post(
    "/step",
    summary="Advance the simulation by step_seconds",
)
async def simulation_step(req: StepRequest):
    """
    Advance the **entire registry** by ``step_seconds`` in one atomic call.

    Pipeline executed on every call:

    | Phase | Action |
    |-------|--------|
    | **1 – Propagation** | RK4-propagate each satellite and debris object. State written back to registry via ``fast_update_*``. |
    | **2 – Maneuvers**   | Execute any ``ScheduledBurn`` records whose ``burn_time`` falls within the current step window. Tsiolkovsky fuel deduction applied. |
    | **3 – Collisions**  | Build KDTree from debris positions, query each satellite. Flag events ≤ ``collision_threshold_km``. |
    | **4 – Clock**       | Advance the simulation clock by ``step_seconds``. |

    Returns
    -------
    ``status``, ``new_timestamp``, ``collisions_detected``, ``maneuvers_executed``
    plus detailed per-object and per-event breakdowns.
    """
    svc = get_step_service()

    # Apply per-request physics overrides
    svc.include_j2             = req.include_j2
    svc.include_drag           = req.include_drag
    svc.collision_threshold_km = req.collision_threshold_km

    try:
        result = svc.step(req.step_seconds)
    except ValueError as exc:
        raise HTTPException(422, str(exc))

    return result.to_dict()


@router.get(
    "/step/clock",
    summary="Get current simulation clock",
)
async def get_step_clock():
    """Return the current simulation clock and step count."""
    svc = get_step_service()
    return {
        "sim_clock":    svc.sim_clock.isoformat(),
        "step_count":   svc._step_count,
        "collision_threshold_km": svc.collision_threshold_km,
        "include_j2":   svc.include_j2,
        "include_drag":  svc.include_drag,
    }


class ClockResetRequest(BaseModel):
    epoch: Optional[str] = Field(
        None,
        description="ISO8601 UTC epoch to reset to (default: now)",
    )


@router.post(
    "/step/reset",
    summary="Reset simulation clock (optionally to a specific epoch)",
)
async def reset_step_clock(req: ClockResetRequest):
    """Reset the simulation clock and step counter.  Does NOT clear the registry."""
    from datetime import datetime
    epoch = None
    if req.epoch:
        try:
            epoch = datetime.fromisoformat(req.epoch.replace("Z", "+00:00")).replace(tzinfo=None)
        except ValueError:
            raise HTTPException(422, f"Invalid epoch format: '{req.epoch}'")
    get_step_service().reset_clock(epoch)
    return {
        "status": "RESET",
        "sim_clock": get_step_service().sim_clock.isoformat(),
    }


@router.get(
    "/step/config",
    summary="Show simulation step service configuration",
)
async def step_config():
    """Return current physics flags and collision threshold for the step service."""
    svc = get_step_service()
    return {
        "collision_threshold_km":  svc.collision_threshold_km,
        "include_j2":              svc.include_j2,
        "include_drag":            svc.include_drag,
        "default_step_seconds":    60.0,
        "max_step_seconds":        86400.0,
        "maneuver_isp_s":          300.0,
        "scheduled_burn_window":   "[sim_clock, sim_clock + step_seconds)",
    }
