"""
maneuver_scheduler.py
=====================
Maneuver Sequence Scheduling Service for the Autonomous Constellation Manager.

Validates and stores sequences of future burns for a satellite before they
are uplinked to the on-board computer or executed by the simulation engine.

Validation checks (per burn, in chronological order)
------------------------------------------------------
1. **Command latency** – burn_time >= now + MIN_LATENCY_S (default 10 s).
   Models the minimum propagation/processing delay for a command uplink.

2. **Cooldown** – burn_time >= previous_burn_time + COOLDOWN_S (600 s).
   Prevents thruster stress from back-to-back firings.  Considers both
   already-scheduled burns (from prior calls) and prior burns in this batch.

3. **Sufficient fuel** – simulate Tsiolkovsky fuel consumption through the
   whole sequence in order.  Each burn sees the post-burn mass from the
   previous step.  If cumulative consumption exceeds usable fuel, the
   burn and all subsequent ones are flagged INVALID.

4. **Ground station LOS** – at each burn epoch, check whether the satellite's
   propagated position is visible from at least one configured ground station.
   Uses a simple horizon model:

       cos θ = (sat_pos · gs_pos) / (|sat_pos| · |gs_pos|)
       In view if θ < 90° + half-Earth-angle

   3 default stations are included (Svalbard, Wallops, Singapore).
   LOS is advisory; the scheduler does not hard-block on LOS failure.

Scheduled burns
---------------
Accepted burns are stored in ``schedule_store[satellite_id]`` sorted by
burn_time.  The store persists for the process lifetime (no persistence to
disk).  Overlapping submissions for the same satellite merge by burn_id
(upsert semantics).

Output schema (per-burn)
------------------------
{
  "burn_id"            : str,
  "burn_time"          : str (ISO8601),
  "delta_v_mag_m_s"    : float,
  "status"             : "SCHEDULED" | "INVALID",
  "validation": {
    "ground_station_los"          : bool,
    "sufficient_fuel"             : bool,
    "cooldown_satisfied"          : bool,
    "latency_satisfied"           : bool,
    "projected_mass_remaining_kg" : float,
    "propellant_used_kg"          : float,
    "rejection_reason"            : str | None,
  }
}
"""

from __future__ import annotations

import logging
import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.physics.fuel_calculator import (
    propellant_mass,
    is_feasible,
    ISP_S, G0_M_S2, MIN_FUEL_KG, FUEL_RESERVE_KG,
)

logger = logging.getLogger(__name__)

# ── Schedule-time constants ───────────────────────────────────────────────────
MIN_LATENCY_S:   float = 10.0    # minimum seconds from 'now' to burn_time
COOLDOWN_S:      float = 600.0   # minimum gap between burns for same satellite
SAT_DRY_MASS_KG: float = 100.0   # default dry-mass for Tsiolkovsky
EARTH_RADIUS_KM: float = 6371.0  # mean Earth radius [km]

# ── Ground station catalogue (ECI ECEF-approximate positions [km]) ────────────
# These are ECEF positions used as a fixed reference.  For a production system
# ECEF→ECI rotation would be applied at each epoch; here we treat them as
# always-valid reference vectors (conservative for scheduling).
GROUND_STATIONS: List[Tuple[str, np.ndarray]] = [
    ("Svalbard",   np.array([ 1252.3,  276.3, 6357.4])),   # 78°N 15°E
    ("Wallops",    np.array([ 1141.5, -5141.8, 3677.6])),   # 37°N 75°W
    ("Singapore",  np.array([-1277.4,  6238.2,  137.9])),   # 1°N 103°E
]


# ═══════════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class BurnValidation:
    """Per-burn validation result."""
    ground_station_los:          bool
    sufficient_fuel:             bool
    cooldown_satisfied:          bool
    latency_satisfied:           bool
    projected_mass_remaining_kg: float
    propellant_used_kg:          float
    rejection_reason:            Optional[str]

    @property
    def all_valid(self) -> bool:
        return (
            self.sufficient_fuel and
            self.cooldown_satisfied and
            self.latency_satisfied
        )

    def to_dict(self) -> dict:
        return {
            "ground_station_los":           self.ground_station_los,
            "sufficient_fuel":              self.sufficient_fuel,
            "cooldown_satisfied":           self.cooldown_satisfied,
            "latency_satisfied":            self.latency_satisfied,
            "projected_mass_remaining_kg":  round(self.projected_mass_remaining_kg, 4),
            "propellant_used_kg":           round(self.propellant_used_kg, 6),
            "rejection_reason":             self.rejection_reason,
        }


@dataclass
class ScheduledBurn:
    """A single validated burn in a satellite's schedule."""
    burn_id:         str
    satellite_id:    str
    burn_time:       datetime       # UTC
    burn_time_iso:   str
    delta_v_vector:  dict           # {"x", "y", "z"} in km/s
    delta_v_mag_m_s: float
    status:          str            # "SCHEDULED" | "INVALID"
    validation:      BurnValidation
    schedule_id:     str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at:      str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict:
        return {
            "schedule_id":     self.schedule_id,
            "burn_id":         self.burn_id,
            "satellite_id":    self.satellite_id,
            "burn_time":       self.burn_time_iso,
            "delta_v_vector":  self.delta_v_vector,
            "delta_v_mag_m_s": round(self.delta_v_mag_m_s, 4),
            "status":          self.status,
            "validation":      self.validation.to_dict(),
            "created_at":      self.created_at,
        }


# ── Module-level schedule store ───────────────────────────────────────────────
# schedule_store[satellite_id] → list[ScheduledBurn] sorted by burn_time
schedule_store: Dict[str, List[ScheduledBurn]] = {}


# ═══════════════════════════════════════════════════════════════════════════
# LOS helper
# ═══════════════════════════════════════════════════════════════════════════

def _check_los(sat_pos_km: np.ndarray) -> Tuple[bool, str]:
    """
    Return (in_view, station_name_or_none) for the first visible ground station.

    Model: satellite is visible from a ground station if the angle between the
    satellite position vector and the ground-station position vector is less than
    90° + the Earth half-angle at the satellite altitude:

        θ_horizon = arcsin(R_e / |sat_pos|)
        visible   = angle(sat_pos, gs_pos) < 90° + θ_horizon

    Equivalently:
        cos(angle) = (sat·gs) / (|sat||gs|)
        visible iff cos(angle) > -sin(θ_horizon) = -R_e / |sat_pos|
    """
    r_sat = float(np.linalg.norm(sat_pos_km))
    if r_sat < 1e-6:
        return False, "none"

    threshold = -EARTH_RADIUS_KM / r_sat   # −sin(θ_horizon)

    for name, gs_pos in GROUND_STATIONS:
        r_gs = float(np.linalg.norm(gs_pos))
        if r_gs < 1e-6:
            continue
        cos_angle = float(np.dot(sat_pos_km, gs_pos)) / (r_sat * r_gs)
        if cos_angle > threshold:
            return True, name

    return False, "none"


# ═══════════════════════════════════════════════════════════════════════════
# Maneuver Scheduler
# ═══════════════════════════════════════════════════════════════════════════

class ManeuverScheduler:
    """
    Validate and store a maneuver sequence for a satellite.

    Parameters
    ----------
    min_latency_s : float  – minimum seconds from 'now' to burn_time
    cooldown_s    : float  – minimum gap between consecutive burns
    dry_mass_kg   : float  – satellite dry mass for Tsiolkovsky
    isp_s         : float  – specific impulse [s]
    """

    def __init__(
        self,
        min_latency_s: float = MIN_LATENCY_S,
        cooldown_s:    float = COOLDOWN_S,
        dry_mass_kg:   float = SAT_DRY_MASS_KG,
        isp_s:         float = ISP_S,
    ) -> None:
        self.min_latency_s = min_latency_s
        self.cooldown_s    = cooldown_s
        self.dry_mass_kg   = dry_mass_kg
        self.isp_s         = isp_s

    # ─── primary entry point ─────────────────────────────────────────────────

    def schedule(
        self,
        satellite_id:      str,
        maneuver_sequence: List[dict],
        epoch:             Optional[datetime] = None,
    ) -> dict:
        """
        Validate and schedule a sequence of burns for a satellite.

        Parameters
        ----------
        satellite_id      : str  – target satellite (must be in registry)
        maneuver_sequence : list[dict]  – each item:
            {
              "burn_id":      str,
              "burnTime":     str (ISO8601),
              "deltaV_vector": {"x": float, "y": float, "z": float}  [km/s]
            }
        epoch : datetime | None  – current UTC time (default: now)

        Returns
        -------
        dict  matching the user-specified response schema.
        """
        from app.data.registry import get_satellite

        epoch = epoch or datetime.utcnow()
        sat   = get_satellite(satellite_id)
        if sat is None:
            raise ValueError(f"Satellite '{satellite_id}' not found in registry.")

        # ── Parse and sort burns chronologically ─────────────────────────────
        raw_burns: List[dict] = []
        for item in maneuver_sequence:
            bt = self._parse_time(item.get("burnTime", ""))
            if bt is None:
                raise ValueError(
                    f"burn_id='{item.get('burn_id', '?')}': "
                    f"invalid burnTime '{item.get('burnTime')}'"
                )
            dv_vec = item.get("deltaV_vector", {"x": 0.0, "y": 0.0, "z": 0.0})
            dv_mag = math.sqrt(
                dv_vec.get("x", 0.0) ** 2 +
                dv_vec.get("y", 0.0) ** 2 +
                dv_vec.get("z", 0.0) ** 2
            ) * 1000.0   # km/s → m/s
            raw_burns.append({
                "burn_id":    item.get("burn_id", str(uuid.uuid4())),
                "burn_time":  bt,
                "dv_vec":     dv_vec,
                "dv_mag_m_s": dv_mag,
            })

        raw_burns.sort(key=lambda b: b["burn_time"])

        # ── Load already-scheduled burns for cooldown reference ───────────────
        existing = schedule_store.get(satellite_id, [])
        # Find the last scheduled burn time from existing store
        last_scheduled_time: Optional[datetime] = None
        for sb in reversed(existing):
            if sb.status == "SCHEDULED":
                last_scheduled_time = sb.burn_time
                break

        # ── Simulate fuel sequentially ────────────────────────────────────────
        fuel_remaining = sat.fuel_kg
        wet_mass       = fuel_remaining + self.dry_mass_kg

        results: List[ScheduledBurn] = []
        prev_burn_time: Optional[datetime] = last_scheduled_time
        fuel_exhausted = False

        for b in raw_burns:
            burn_id    = b["burn_id"]
            burn_time  = b["burn_time"]
            dv_mag_m_s = b["dv_mag_m_s"]
            dv_vec     = b["dv_vec"]

            rejection: Optional[str] = None

            # 1. Latency check
            delta_t = (burn_time - epoch).total_seconds()
            latency_ok = delta_t >= self.min_latency_s
            if not latency_ok:
                rejection = (
                    f"Command latency {delta_t:.1f} s < "
                    f"minimum {self.min_latency_s} s"
                )

            # 2. Cooldown check
            if prev_burn_time is not None:
                gap_s     = (burn_time - prev_burn_time).total_seconds()
                cooldown_ok = gap_s >= self.cooldown_s
                if not cooldown_ok and rejection is None:
                    rejection = (
                        f"Cooldown violation: {gap_s:.0f} s gap < "
                        f"{self.cooldown_s} s required"
                    )
            else:
                cooldown_ok = True

            # 3. Fuel feasibility
            if fuel_exhausted:
                fuel_ok    = False
                prop_used  = 0.0
                if rejection is None:
                    rejection  = "Fuel exhausted by prior maneuvers in sequence"
            elif dv_mag_m_s <= 0.0:
                fuel_ok   = True
                prop_used = 0.0
            else:
                ok, required, reason = is_feasible(
                    fuel_remaining, wet_mass, dv_mag_m_s, self.isp_s
                )
                fuel_ok   = ok
                prop_used = required if ok else 0.0
                if not ok:
                    fuel_exhausted = True
                    if rejection is None:
                        rejection = reason

            # Consume fuel even on partial failure to propagate state
            if fuel_ok and dv_mag_m_s > 0.0:
                prop_used  = propellant_mass(wet_mass, dv_mag_m_s, self.isp_s)
                fuel_remaining = max(0.0, fuel_remaining - prop_used)
                wet_mass       = fuel_remaining + self.dry_mass_kg

            # 4. LOS check
            sat_pos = np.array([sat.position.x, sat.position.y, sat.position.z])
            los_ok, los_station = _check_los(sat_pos)

            # Determine overall status
            burn_valid = latency_ok and cooldown_ok and fuel_ok
            status     = "SCHEDULED" if burn_valid else "INVALID"

            validation = BurnValidation(
                ground_station_los          = los_ok,
                sufficient_fuel             = fuel_ok,
                cooldown_satisfied          = cooldown_ok,
                latency_satisfied           = latency_ok,
                projected_mass_remaining_kg = wet_mass,
                propellant_used_kg          = prop_used,
                rejection_reason            = rejection,
            )

            sb = ScheduledBurn(
                burn_id         = burn_id,
                satellite_id    = satellite_id,
                burn_time       = burn_time,
                burn_time_iso   = burn_time.isoformat(),
                delta_v_vector  = dv_vec,
                delta_v_mag_m_s = dv_mag_m_s,
                status          = status,
                validation      = validation,
            )
            results.append(sb)

            if burn_valid:
                prev_burn_time = burn_time

        # ── Upsert valid burns into schedule store ────────────────────────────
        self._upsert_to_store(satellite_id, results)

        # ── Build response ────────────────────────────────────────────────────
        all_valid        = all(sb.status == "SCHEDULED" for sb in results)
        any_invalid      = any(sb.status == "INVALID"   for sb in results)
        overall_status   = "SCHEDULED" if all_valid else ("PARTIAL" if any_invalid and not all_valid else "REJECTED")
        scheduled_burns  = [sb for sb in results if sb.status == "SCHEDULED"]
        invalid_burns    = [sb for sb in results if sb.status == "INVALID"]

        # Aggregate validation flags
        agg_validation = {
            "ground_station_los":          any(sb.validation.ground_station_los for sb in results),
            "sufficient_fuel":             all(sb.validation.sufficient_fuel for sb in results),
            "cooldown_satisfied":          all(sb.validation.cooldown_satisfied for sb in results),
            "latency_satisfied":           all(sb.validation.latency_satisfied for sb in results),
            "projected_mass_remaining_kg": round(fuel_remaining, 4),
            "total_propellant_used_kg":    round(
                sum(sb.validation.propellant_used_kg for sb in results), 6
            ),
            "burns_scheduled": len(scheduled_burns),
            "burns_invalid":   len(invalid_burns),
        }

        return {
            "status":              overall_status,
            "satellite_id":        satellite_id,
            "total_burns":         len(results),
            "validation":          agg_validation,
            "scheduled_at":        epoch.isoformat(),
            "maneuver_sequence":   [sb.to_dict() for sb in results],
        }

    # ─── query the schedule store ─────────────────────────────────────────────

    def get_schedule(self, satellite_id: str) -> List[dict]:
        """Return all stored burns for a satellite."""
        burns = schedule_store.get(satellite_id, [])
        return [sb.to_dict() for sb in burns]

    def cancel_burn(self, satellite_id: str, burn_id: str) -> bool:
        """Mark a stored burn as CANCELLED."""
        for sb in schedule_store.get(satellite_id, []):
            if sb.burn_id == burn_id and sb.status == "SCHEDULED":
                sb.status = "CANCELLED"
                return True
        return False

    def clear_schedule(self, satellite_id: str) -> int:
        """Remove all scheduled burns for a satellite. Returns count removed."""
        n = len(schedule_store.get(satellite_id, []))
        schedule_store.pop(satellite_id, None)
        return n

    # ─── helpers ─────────────────────────────────────────────────────────────

    def _parse_time(self, t: str) -> Optional[datetime]:
        """Parse ISO8601 datetime string → naive UTC datetime."""
        if not t:
            return None
        for fmt in (
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
        ):
            try:
                return datetime.strptime(t, fmt).replace(tzinfo=None)
            except ValueError:
                continue
        # Try stdlib fromisoformat (Python 3.11+ handles Z suffix)
        try:
            dt = datetime.fromisoformat(t.replace("Z", "+00:00"))
            return dt.replace(tzinfo=None)
        except Exception:
            return None

    def _upsert_to_store(
        self, satellite_id: str, new_burns: List[ScheduledBurn]
    ) -> None:
        """Merge new burns into the schedule store (upsert by burn_id)."""
        existing = {sb.burn_id: sb for sb in schedule_store.get(satellite_id, [])}
        for nb in new_burns:
            existing[nb.burn_id] = nb   # overwrite on re-schedule
        schedule_store[satellite_id] = sorted(
            existing.values(), key=lambda x: x.burn_time
        )


# ═══════════════════════════════════════════════════════════════════════════
# Module-level singleton
# ═══════════════════════════════════════════════════════════════════════════

_scheduler: Optional[ManeuverScheduler] = None


def get_scheduler() -> ManeuverScheduler:
    global _scheduler
    if _scheduler is None:
        _scheduler = ManeuverScheduler()
    return _scheduler
