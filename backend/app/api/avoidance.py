"""
Collision Avoidance Maneuver API
=================================
Endpoints that expose the AvoidancePlanner to operators.

Endpoints
---------
  POST /api/v1/avoidance/plan
      Plan burns for all CDMs currently in the conjunction service store,
      filtered by risk level.  Returns a list of ManeuverPlan objects.

  POST /api/v1/avoidance/plan-cdm
      Supply CDM payloads inline (no prior /cdm/run needed).

  GET  /api/v1/avoidance/cooldowns
      Show per-satellite cooldown expiry times.

  DELETE /api/v1/avoidance/cooldowns
      Reset all cooldown records (simulation / testing).

  GET  /api/v1/avoidance/constraints
      Return planner constraints (MAX_DV, COOLDOWN, MIN_FUEL, etc.).

  POST /api/v1/avoidance/configure
      Reconfigure the shared planner (risk threshold, lead time).
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.services.avoidance_planner import (
    get_planner,
    reset_planner,
    ManeuverPlan,
    MAX_DV_KM_S, COOLDOWN_S, SAFE_SEP_KM,
    BURN_LEAD_TIME_S, SAT_DRY_MASS_KG,
)
from app.physics.fuel_calculator import (
    propellant_mass,
    mass_after_burn,
    max_delta_v_m_s,
    is_feasible as _fuel_is_feasible,
    fuel_budget_summary,
    apply_burn_to_registry,
    InsufficientFuelError,
    burn_ledger,
    ISP_S, G0_M_S2, MIN_FUEL_KG, FUEL_RESERVE_KG,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/avoidance")


# ═══════════════════════════════════════════════════════════════════════════
# Request schemas
# ═══════════════════════════════════════════════════════════════════════════

class PlanRequest(BaseModel):
    """Parameters for a planning run against stored CDMs."""
    risk_threshold: str   = Field("WARNING",
                                   description="Minimum risk level to plan: CRITICAL|WARNING|CAUTION")
    lead_time_s:    float = Field(BURN_LEAD_TIME_S, gt=0, le=3600,
                                   description="Seconds before TCA to fire burn [s]")
    feasible_only:  bool  = Field(False,
                                   description="Return only feasible burn plans")


class CDMInlinePayload(BaseModel):
    """An inline CDM for ad-hoc planning without a prior /cdm/run call."""
    satellite_id:             str
    debris_id:                str
    satellite_name:           str  = "unknown"
    debris_designation:       str  = "unknown"
    time_of_closest_approach: str
    tca_offset_s:             float
    miss_distance_km:         float
    relative_speed_km_s:      float
    risk_level:               str
    collision_probability:    float = 0.0
    sat_pos_at_tca:           List[float]  = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    deb_pos_at_tca:           List[float]  = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    generated_at:             str  = ""


class PlanInlineRequest(BaseModel):
    cdms:           List[CDMInlinePayload]
    risk_threshold: str   = Field("WARNING")
    lead_time_s:    float = Field(BURN_LEAD_TIME_S, gt=0)
    feasible_only:  bool  = Field(False)


class ConfigureRequest(BaseModel):
    risk_threshold: str   = Field("WARNING")
    lead_time_s:    float = Field(BURN_LEAD_TIME_S, gt=0, le=3600)


# ═══════════════════════════════════════════════════════════════════════════
# Endpoints
# ═══════════════════════════════════════════════════════════════════════════

@router.get("/constraints", summary="Planner constraint constants")
async def get_constraints():
    """Return the fixed and configurable constraints of the avoidance planner."""
    return {
        "max_delta_v_m_s":    MAX_DV_KM_S * 1000.0,
        "max_delta_v_km_s":   MAX_DV_KM_S,
        "cooldown_seconds":   COOLDOWN_S,
        "min_fuel_kg":        MIN_FUEL_KG,
        "fuel_reserve_kg":    FUEL_RESERVE_KG,
        "safe_separation_km": SAFE_SEP_KM,
        "default_lead_time_s": BURN_LEAD_TIME_S,
        "isp_s":              ISP_S,
        "g0_m_s2":            G0_M_S2,
        "formula":            "Δm = m · (1 − exp(−Δv / (Isp · g₀)))",
        "risk_levels":        ["CRITICAL", "WARNING", "CAUTION", "NOMINAL"],
    }


@router.post("/configure", summary="Reconfigure the shared avoidance planner")
async def configure(req: ConfigureRequest):
    planner = reset_planner(
        risk_threshold = req.risk_threshold,
        lead_time_s    = req.lead_time_s,
    )
    return {
        "status":         "reconfigured",
        "risk_threshold": planner.risk_threshold,
        "lead_time_s":    planner.lead_time_s,
    }


@router.get("/cooldowns", summary="Show per-satellite cooldown expiry times")
async def get_cooldowns():
    planner = get_planner()
    return {
        "cooldowns": {
            sid: dt.isoformat()
            for sid, dt in planner._cooldown_until.items()
        }
    }


@router.delete("/cooldowns", status_code=204, summary="Reset all cooldown records")
async def reset_cooldowns():
    get_planner().reset_cooldowns()


@router.post(
    "/plan",
    summary="Plan avoidance burns for stored CDMs",
)
async def plan_from_store(req: PlanRequest):
    """
    Run the avoidance planner against all CDMs currently stored in the
    conjunction service.  Requires a prior ``POST /cdm/run`` call.

    Returns a list of ManeuverPlan objects (burn_id, burn_time, ΔV vector)
    sorted by risk level descending.
    """
    from app.services.conjunction_service import get_service

    svc = get_service()
    if not svc.last_run_at:
        raise HTTPException(
            400,
            "No conjunction data available. Run POST /api/v1/cdm/run first."
        )

    planner = get_planner()
    planner.risk_threshold = req.risk_threshold.upper()
    planner.lead_time_s    = req.lead_time_s

    cdms  = svc.get_cdms()
    epoch = datetime.utcnow()
    plans = planner.plan(cdms, epoch=epoch)

    if req.feasible_only:
        plans = [p for p in plans if p.feasible]

    feasible_count   = sum(1 for p in plans if p.feasible)
    infeasible_count = len(plans) - feasible_count

    return {
        "epoch":            epoch.isoformat(),
        "total_plans":      len(plans),
        "feasible_burns":   feasible_count,
        "infeasible_burns": infeasible_count,
        "maneuver_plans":   [p.to_dict() for p in plans],
    }


@router.post(
    "/plan-cdm",
    summary="Plan avoidance burns from inline CDM payloads",
)
async def plan_from_inline(req: PlanInlineRequest):
    """
    Supply CDMs directly in the request body without depending on
    a prior conjunction run.  Useful for testing and external integrations.
    """
    from app.services.conjunction_service import ConjunctionDataMessage

    cdms = [
        ConjunctionDataMessage(
            satellite_id              = c.satellite_id,
            debris_id                 = c.debris_id,
            satellite_name            = c.satellite_name,
            debris_designation        = c.debris_designation,
            time_of_closest_approach  = c.time_of_closest_approach,
            tca_offset_s              = c.tca_offset_s,
            miss_distance_km          = c.miss_distance_km,
            relative_speed_km_s       = c.relative_speed_km_s,
            risk_level                = c.risk_level,
            collision_probability     = c.collision_probability,
            sat_pos_at_tca            = c.sat_pos_at_tca,
            deb_pos_at_tca            = c.deb_pos_at_tca,
            generated_at              = c.generated_at or datetime.utcnow().isoformat(),
        )
        for c in req.cdms
    ]

    planner = get_planner()
    planner.risk_threshold = req.risk_threshold.upper()
    planner.lead_time_s    = req.lead_time_s

    epoch = datetime.utcnow()
    plans = planner.plan(cdms, epoch=epoch)

    if req.feasible_only:
        plans = [p for p in plans if p.feasible]

    return {
        "epoch":          epoch.isoformat(),
        "total_plans":    len(plans),
        "feasible_burns": sum(1 for p in plans if p.feasible),
        "maneuver_plans": [p.to_dict() for p in plans],
    }


# ════════════════════════════════════════════════════════════════════════
# Fuel calculator endpoints
# ════════════════════════════════════════════════════════════════════════

class FuelCalcRequest(BaseModel):
    current_mass_kg: float  = Field(...,  gt=0,  description="Current wet mass [kg]")
    delta_v_m_s:     float  = Field(...,  ge=0,  description="ΔV magnitude [m/s]")
    isp_s:           float  = Field(ISP_S, gt=0, description="Specific impulse [s]")


@router.post(
    "/fuel/calculate",
    summary="Calculate propellant consumption (Tsiolkovsky)",
)
async def calculate_fuel(req: FuelCalcRequest):
    """
    Apply the Tsiolkovsky rocket equation with the configured Isp.

        Δm = m · (1 − exp(−Δv / (Isp · g₀)))

    Parameters accepted in the request body:
    - ``current_mass_kg``  – total wet mass before the burn
    - ``delta_v_m_s``      – ΔV magnitude [m/s]
    - ``isp_s``            – specific impulse [s] (default 300 s)
    """
    delta_m     = propellant_mass(req.current_mass_kg, req.delta_v_m_s, req.isp_s)
    mass_after  = req.current_mass_kg - delta_m
    v_e         = req.isp_s * G0_M_S2
    return {
        "current_mass_kg":   req.current_mass_kg,
        "delta_v_m_s":       req.delta_v_m_s,
        "isp_s":             req.isp_s,
        "g0_m_s2":           G0_M_S2,
        "exhaust_velocity_m_s": round(v_e, 4),
        "propellant_consumed_kg": round(delta_m, 6),
        "mass_after_kg":     round(mass_after, 4),
        "fuel_fraction":     round(delta_m / req.current_mass_kg, 8),
        "formula":           "Δm = m · (1 − exp(−Δv / (Isp · g₀)))",
    }


@router.get(
    "/fuel/{satellite_id}",
    summary="Fuel budget summary for a registry satellite",
)
async def get_fuel_budget(
    satellite_id: str,
    dry_mass_kg:  float = Query(SAT_DRY_MASS_KG, gt=0, description="Satellite dry mass [kg]"),
    isp_s:        float = Query(ISP_S, gt=0, description="Specific impulse [s]"),
):
    """
    Return full fuel budget for a registered satellite:
    fuel remaining, usable budget (after reserve), and maximum ΔV
    achievable with current propellant.
    """
    try:
        return fuel_budget_summary(satellite_id, dry_mass_kg=dry_mass_kg, isp_s=isp_s)
    except KeyError as e:
        raise HTTPException(404, str(e))


@router.post(
    "/fuel/check",
    summary="Check burn feasibility for a registry satellite",
)
async def check_feasibility(
    satellite_id: str,
    delta_v_m_s:  float = Query(..., gt=0, description="Requested ΔV [m/s]"),
    dry_mass_kg:  float = Query(SAT_DRY_MASS_KG, gt=0),
    isp_s:        float = Query(ISP_S, gt=0),
):
    """
    Check whether a satellite has enough fuel to execute a given ΔV.
    Returns feasibility flag, required propellant, and remaining fuel after burn.
    """
    from app.data.registry import get_satellite
    sat = get_satellite(satellite_id)
    if sat is None:
        raise HTTPException(404, f"Satellite '{satellite_id}' not found")

    wet_mass = sat.fuel_kg + dry_mass_kg
    ok, required, reason = _fuel_is_feasible(sat.fuel_kg, wet_mass, delta_v_m_s, isp_s)
    return {
        "satellite_id":   satellite_id,
        "fuel_kg":        round(sat.fuel_kg, 4),
        "wet_mass_kg":    round(wet_mass, 4),
        "delta_v_m_s":    delta_v_m_s,
        "isp_s":          isp_s,
        "feasible":       ok,
        "propellant_required_kg": round(required, 6),
        "fuel_remaining_kg": round(max(0.0, sat.fuel_kg - required), 4) if ok else round(sat.fuel_kg, 4),
        "reason":         reason,
    }


# ════════════════════════════════════════════════════════════════════════
# Burn execution  (committing a plan to the registry)
# ════════════════════════════════════════════════════════════════════════

class ExecuteRequest(BaseModel):
    """Burn execution request – supply a ManeuverPlan dict to commit."""
    satellite_id:     str
    burn_id:          str
    delta_v_mag_m_s:  float = Field(..., gt=0, le=MAX_DV_KM_S * 1000.0 + 1e-6)
    delta_v_km_s:     dict  = Field(
        default_factory=lambda: {"x": 0.0, "y": 0.0, "z": 0.0},
        description="ECI ΔV vector [km/s] to add to satellite velocity",
    )
    dry_mass_kg:      float = Field(SAT_DRY_MASS_KG, gt=0)
    isp_s:            float = Field(ISP_S, gt=0)
    infeasible_reason: Optional[str] = None
    feasible:          bool = True


@router.post(
    "/execute",
    summary="Execute a maneuver burn – deduct fuel and update velocity",
)
async def execute_burn(req: ExecuteRequest):
    """
    Commit a maneuver burn to the registry:

    1. Re-validate fuel feasibility with the **Tsiolkovsky formula** (Isp=300 s).
    2. Compute exact propellant mass:  ``Δm = m · (1 − exp(−Δv / (Isp · g₀)))``.
    3. Deduct ``Δm`` from ``satellite.fuel_kg`` in the registry.
    4. Add the ΔV vector to ``satellite.velocity``.
    5. Return a ``BurnRecord`` with full before/after accounting.

    The endpoint **refuses** to execute if:
    - The satellite has < MIN_FUEL_KG remaining
    - Required propellant exceeds usable budget (fuel − FUEL_RESERVE_KG)
    - The satellite is not in the registry
    """
    if not req.feasible:
        raise HTTPException(400, f"Plan is marked infeasible: {req.infeasible_reason}")

    from app.data.registry import get_satellite, fast_update_satellite

    sat = get_satellite(req.satellite_id)
    if sat is None:
        raise HTTPException(404, f"Satellite '{req.satellite_id}' not found")

    try:
        record = apply_burn_to_registry(
            satellite_id = req.satellite_id,
            burn_id      = req.burn_id,
            delta_v_m_s  = req.delta_v_mag_m_s,
            dry_mass_kg  = req.dry_mass_kg,
            isp_s        = req.isp_s,
        )
    except InsufficientFuelError as e:
        raise HTTPException(409, str(e))
    except KeyError as e:
        raise HTTPException(404, str(e))

    # Apply ΔV to satellite velocity
    dv = req.delta_v_km_s
    fast_update_satellite(
        req.satellite_id,
        vx=sat.velocity.vx + float(dv.get("x", 0.0)),
        vy=sat.velocity.vy + float(dv.get("y", 0.0)),
        vz=sat.velocity.vz + float(dv.get("z", 0.0)),
    )

    return {
        "status":              "executed",
        "satellite_id":        req.satellite_id,
        "burn_id":             req.burn_id,
        "delta_v_mag_m_s":     req.delta_v_mag_m_s,
        "propellant_used_kg":  round(record.propellant_used, 6),
        "fuel_before_kg":      round(record.fuel_before, 4),
        "fuel_after_kg":       round(record.fuel_after, 4),
        "burn_record":         record.to_dict(),
    }


@router.get(
    "/ledger",
    summary="View the burn execution audit ledger",
)
async def get_burn_ledger(
    satellite_id: Optional[str] = Query(None, description="Filter by satellite ID"),
    limit:        int            = Query(100, ge=1, le=1000),
):
    """
    Return all executed burn records from the in-memory ledger.
    Records are in chronological order (most recent last).
    """
    records = list(reversed(burn_ledger))   # most recent first
    if satellite_id:
        records = [r for r in records if r.satellite_id == satellite_id]
    records = records[:limit]
    return {
        "total_burns":   len(burn_ledger),
        "shown":         len(records),
        "records":       [r.to_dict() for r in records],
    }
