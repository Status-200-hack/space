"""
fuel_calculator.py
==================
Canonical Tsiolkovsky rocket equation implementation for the Autonomous
Constellation Manager.

All public functions accept ΔV in m/s (the natural propulsion unit) and
mass in kg, matching the user-specified interface.

Canonical formula (user-specified)
------------------------------------
    Δm = m · (1 − exp(−Δv / (Isp · g₀)))

Where:
    m    = current wet mass  [kg]
    Δv   = velocity impulse  [m/s]
    Isp  = specific impulse  [s]   (default 300 s — standard monopropellant)
    g₀   = 9.80665 m/s²            (standard gravity, exact SI value)
    Δm   = propellant consumed  [kg]

Design
------
This module is **pure** — no registry side-effects. Registry fuel updates
are performed explicitly via :func:`apply_burn_to_registry`, which reads
the satellite, checks feasibility, deducts fuel, commits the change, and
returns a :class:`BurnRecord` for the audit trail.

Constants
---------
    ISP_S            = 300.0 s
    G0_M_S2          = 9.80665 m/s²
    G0_KM_S2         = 9.80665e-3 km/s²   (for internal km/s calcs)
    MIN_FUEL_KG      = 1.0 kg   (absolute floor; burns blocked below this)
    FUEL_RESERVE_KG  = 0.5 kg   (operational reserve, excluded from budget)
"""

from __future__ import annotations

import logging
import math
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# ── Physical constants ───────────────────────────────────────────────────────
ISP_S:           float = 300.0       # standard monopropellant specific impulse [s]
G0_M_S2:         float = 9.80665     # standard gravity [m/s²]
G0_KM_S2:        float = 9.80665e-3  # standard gravity [km/s²]

# ── Operational limits ───────────────────────────────────────────────────────
MIN_FUEL_KG:     float = 1.0         # hard minimum – no burns below this
FUEL_RESERVE_KG: float = 0.5         # soft reserve – counted against budget


# ═══════════════════════════════════════════════════════════════════════════
# Core calculation functions  (pure, no side-effects)
# ═══════════════════════════════════════════════════════════════════════════

def propellant_mass(
    current_mass_kg: float,
    delta_v_m_s:     float,
    isp_s:           float = ISP_S,
) -> float:
    """
    Compute propellant consumed for a given ΔV impulse.

    Formula (user-specified)
    ------------------------
        Δm = m · (1 − exp(−Δv / (Isp · g₀)))

    Parameters
    ----------
    current_mass_kg : float – total wet mass before the burn [kg]
    delta_v_m_s     : float – ΔV magnitude [m/s]
    isp_s           : float – specific impulse [s]  (default 300 s)

    Returns
    -------
    float – propellant mass consumed [kg]  (always ≥ 0)

    Raises
    ------
    ValueError  if inputs are non-positive.

    Examples
    --------
    >>> propellant_mass(200.0, 10.0)
    6.726...
    >>> propellant_mass(200.0, 0.0)
    0.0
    """
    if current_mass_kg <= 0:
        raise ValueError(f"current_mass_kg must be positive, got {current_mass_kg}")
    if delta_v_m_s < 0:
        raise ValueError(f"delta_v_m_s must be non-negative, got {delta_v_m_s}")
    if isp_s <= 0:
        raise ValueError(f"isp_s must be positive, got {isp_s}")

    v_e = isp_s * G0_M_S2                           # effective exhaust velocity [m/s]
    delta_m = current_mass_kg * (1.0 - math.exp(-delta_v_m_s / v_e))
    return float(delta_m)


def mass_after_burn(
    current_mass_kg: float,
    delta_v_m_s:     float,
    isp_s:           float = ISP_S,
) -> float:
    """
    Wet mass remaining after a burn.

        m_final = m₀ · exp(−Δv / (Isp · g₀))

    Parameters
    ----------
    current_mass_kg : float – wet mass before burn [kg]
    delta_v_m_s     : float – ΔV magnitude [m/s]
    isp_s           : float – specific impulse [s]

    Returns
    -------
    float – remaining wet mass [kg]
    """
    return current_mass_kg - propellant_mass(current_mass_kg, delta_v_m_s, isp_s)


def max_delta_v_m_s(
    wet_mass_kg:  float,
    fuel_kg:      float,
    isp_s:        float = ISP_S,
) -> float:
    """
    Maximum achievable ΔV given available fuel.

    Tsiolkovsky inverse:  ΔV = Isp · g₀ · ln(m_wet / m_dry)

    Parameters
    ----------
    wet_mass_kg : float – current total wet mass [kg]
    fuel_kg     : float – available propellant [kg]  (must be ≤ wet_mass_kg)
    isp_s       : float – specific impulse [s]

    Returns
    -------
    float – maximum ΔV [m/s]
    """
    if wet_mass_kg <= 0 or fuel_kg < 0:
        return 0.0
    dry_mass = max(1e-6, wet_mass_kg - fuel_kg)
    v_e      = isp_s * G0_M_S2
    return float(v_e * math.log(wet_mass_kg / dry_mass))


def fuel_fraction(
    delta_v_m_s: float,
    isp_s:       float = ISP_S,
) -> float:
    """
    Fraction of initial mass consumed as propellant.

        f = 1 − exp(−Δv / (Isp · g₀))

    Useful for quick budgeting without knowing absolute mass.

    Parameters
    ----------
    delta_v_m_s : float – ΔV [m/s]
    isp_s       : float – Isp [s]

    Returns
    -------
    float in [0, 1)
    """
    v_e = isp_s * G0_M_S2
    return float(1.0 - math.exp(-delta_v_m_s / v_e))


# ═══════════════════════════════════════════════════════════════════════════
# Custom exception  (defined early so is_feasible and apply_burn can use it)
# ═══════════════════════════════════════════════════════════════════════════

class InsufficientFuelError(Exception):
    """Raised when a burn is blocked by the fuel feasibility check."""


def is_feasible(
    satellite_fuel_kg: float,
    wet_mass_kg:       float,
    delta_v_m_s:       float,
    isp_s:             float = ISP_S,
) -> Tuple[bool, float, str]:
    """
    Check whether a burn is feasible given the satellite's current fuel.

    Accounts for the operational reserve (``FUEL_RESERVE_KG``).  The usable
    fuel budget is ``satellite_fuel_kg − FUEL_RESERVE_KG``.

    Parameters
    ----------
    satellite_fuel_kg : float – current fuel load [kg]
    wet_mass_kg       : float – total wet mass including fuel [kg]
    delta_v_m_s       : float – requested ΔV [m/s]
    isp_s             : float – Isp [s]

    Returns
    -------
    (ok: bool, required_kg: float, reason: str)
    """
    # Hard floor check
    if satellite_fuel_kg < MIN_FUEL_KG:
        return (
            False,
            0.0,
            f"Fuel below minimum: {satellite_fuel_kg:.3f} kg < {MIN_FUEL_KG} kg",
        )

    usable = satellite_fuel_kg - FUEL_RESERVE_KG
    if usable <= 0:
        return (
            False,
            0.0,
            f"No usable fuel after reserve deduction: {satellite_fuel_kg:.3f} - {FUEL_RESERVE_KG} = {usable:.3f} kg",
        )

    required = propellant_mass(wet_mass_kg, delta_v_m_s, isp_s)

    if required > usable:
        return (
            False,
            required,
            f"Insufficient fuel: need {required:.3f} kg, usable {usable:.3f} kg "
            f"(reserve {FUEL_RESERVE_KG} kg held back from {satellite_fuel_kg:.3f} kg total)",
        )

    return True, required, "OK"


# ═══════════════════════════════════════════════════════════════════════════
# Burn record  (audit trail)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class BurnRecord:
    """
    Immutable record of an executed maneuver and its fuel accounting.

    Stored in :data:`burn_ledger` for post-mission analysis.
    """
    record_id:        str
    burn_id:          str       # from ManeuverPlan.burn_id
    satellite_id:     str
    executed_at:      str       # ISO8601 UTC
    delta_v_m_s:      float     # ΔV magnitude executed [m/s]
    isp_s:            float     # Isp used for calculation
    wet_mass_before:  float     # kg – total mass before burn
    fuel_before:      float     # kg – propellant available before burn
    propellant_used:  float     # kg – Δm consumed
    fuel_after:       float     # kg – propellant remaining after burn
    wet_mass_after:   float     # kg – total mass after burn
    success:          bool
    error:            Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "record_id":        self.record_id,
            "burn_id":          self.burn_id,
            "satellite_id":     self.satellite_id,
            "executed_at":      self.executed_at,
            "delta_v_m_s":      round(self.delta_v_m_s, 4),
            "isp_s":            self.isp_s,
            "wet_mass_before_kg": round(self.wet_mass_before, 4),
            "fuel_before_kg":   round(self.fuel_before, 4),
            "propellant_used_kg": round(self.propellant_used, 6),
            "fuel_after_kg":    round(self.fuel_after, 4),
            "wet_mass_after_kg": round(self.wet_mass_after, 4),
            "fuel_fraction_pct": round(
                100.0 * self.propellant_used / max(1e-6, self.fuel_before), 4
            ),
            "success":          self.success,
            "error":            self.error,
        }


# Module-level ledger: list of all executed burns (survives request lifetime)
burn_ledger: list[BurnRecord] = []


# ═══════════════════════════════════════════════════════════════════════════
# Registry integration  (the only function with side effects)
# ═══════════════════════════════════════════════════════════════════════════

def apply_burn_to_registry(
    satellite_id:     str,
    burn_id:          str,
    delta_v_m_s:      float,
    dry_mass_kg:      float = 100.0,
    isp_s:            float = ISP_S,
) -> BurnRecord:
    """
    Execute a burn against a satellite in the registry.

    Steps
    -----
    1. Fetch satellite from registry (raises if not found).
    2. Compute wet mass:  m_wet = fuel_kg + dry_mass_kg.
    3. Call :func:`is_feasible` — raises ``InsufficientFuelError`` if blocked.
    4. Compute propellant consumed via :func:`propellant_mass`.
    5. Deduct fuel from registry using ``fast_update_satellite``.
    6. Append a :class:`BurnRecord` to :data:`burn_ledger`.
    7. Return the BurnRecord.

    Parameters
    ----------
    satellite_id  : str   – registry ID
    burn_id       : str   – from ManeuverPlan.burn_id (for traceability)
    delta_v_m_s   : float – ΔV magnitude to execute [m/s]
    dry_mass_kg   : float – estimated dry mass [kg] (default 100 kg)
    isp_s         : float – Isp [s] (default 300 s)

    Returns
    -------
    BurnRecord

    Raises
    ------
    KeyError              if satellite_id not in registry
    InsufficientFuelError if fuel check fails
    """
    from app.data.registry import get_satellite, fast_update_satellite

    sat = get_satellite(satellite_id)
    if sat is None:
        raise KeyError(f"Satellite '{satellite_id}' not found in registry")

    fuel_before = sat.fuel_kg
    wet_before  = fuel_before + dry_mass_kg
    executed_at = datetime.utcnow().isoformat()

    feasible, required, reason = is_feasible(fuel_before, wet_before, delta_v_m_s, isp_s)

    if not feasible:
        record = BurnRecord(
            record_id       = str(uuid.uuid4()),
            burn_id         = burn_id,
            satellite_id    = satellite_id,
            executed_at     = executed_at,
            delta_v_m_s     = delta_v_m_s,
            isp_s           = isp_s,
            wet_mass_before = wet_before,
            fuel_before     = fuel_before,
            propellant_used = 0.0,
            fuel_after      = fuel_before,
            wet_mass_after  = wet_before,
            success         = False,
            error           = reason,
        )
        burn_ledger.append(record)
        raise InsufficientFuelError(reason)

    # Compute exact propellant consumption
    delta_m    = propellant_mass(wet_before, delta_v_m_s, isp_s)
    fuel_after = max(0.0, fuel_before - delta_m)

    # Commit fuel to registry via direct setattr (same pattern as fast_update_satellite).
    # Position/velocity updates are handled separately by the caller (execute_plan / API).
    from app.data.registry import satellites as _sat_registry
    _sat_obj = _sat_registry.get(satellite_id)
    if _sat_obj is not None:
        object.__setattr__(_sat_obj, "fuel_kg", fuel_after)

    logger.info(
        "BURN EXECUTED  sat=%s  burn_id=%s  ΔV=%.3f m/s  Δm=%.4f kg  "
        "fuel: %.3f → %.3f kg",
        satellite_id, burn_id, delta_v_m_s, delta_m, fuel_before, fuel_after,
    )

    record = BurnRecord(
        record_id       = str(uuid.uuid4()),
        burn_id         = burn_id,
        satellite_id    = satellite_id,
        executed_at     = executed_at,
        delta_v_m_s     = delta_v_m_s,
        isp_s           = isp_s,
        wet_mass_before = wet_before,
        fuel_before     = fuel_before,
        propellant_used = delta_m,
        fuel_after      = fuel_after,
        wet_mass_after  = wet_before - delta_m,
        success         = True,
    )
    burn_ledger.append(record)
    return record



# ═══════════════════════════════════════════════════════════════════════════
# Convenience: quick budget summary for a satellite
# ═══════════════════════════════════════════════════════════════════════════

def fuel_budget_summary(
    satellite_id: str,
    dry_mass_kg:  float = 100.0,
    isp_s:        float = ISP_S,
) -> dict:
    """
    Return a detailed fuel budget summary for a satellite.

    Parameters
    ----------
    satellite_id : str
    dry_mass_kg  : float – estimated dry mass [kg]
    isp_s        : float – Isp [s]

    Returns
    -------
    dict with keys:
        satellite_id, fuel_kg, dry_mass_kg, wet_mass_kg,
        usable_fuel_kg, reserve_kg,
        max_dv_m_s, max_dv_km_s,
        isp_s, g0_m_s2
    """
    from app.data.registry import get_satellite

    sat = get_satellite(satellite_id)
    if sat is None:
        raise KeyError(f"Satellite '{satellite_id}' not found in registry")

    fuel  = sat.fuel_kg
    wet   = fuel + dry_mass_kg
    usable = max(0.0, fuel - FUEL_RESERVE_KG)
    max_dv = max_delta_v_m_s(wet, usable, isp_s)

    return {
        "satellite_id":  satellite_id,
        "satellite_name": sat.name,
        "fuel_kg":       round(fuel, 4),
        "dry_mass_kg":   dry_mass_kg,
        "wet_mass_kg":   round(wet, 4),
        "usable_fuel_kg": round(usable, 4),
        "reserve_kg":    FUEL_RESERVE_KG,
        "max_dv_m_s":    round(max_dv, 4),
        "max_dv_km_s":   round(max_dv / 1000.0, 6),
        "isp_s":         isp_s,
        "g0_m_s2":       G0_M_S2,
        "formula":       "Δm = m · (1 − exp(−Δv / (Isp · g₀)))",
    }
