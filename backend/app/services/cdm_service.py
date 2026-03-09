"""
CDM (Conjunction Data Message) Service.

Screens every object pair in the registry for close approaches using
a fast Cartesian range check and maintains a live, deduplicated set of
active CDM warnings (keyed by object-pair).

Design goals
------------
• O(N²) pair scan but runs in pure Python with no propagation overhead —
  uses instantaneous ECI positions from the registry.
• Results are cached in a module-level dict so the HTTP handler can read
  the count in O(1) without re-running the scan.
• Auto-expires warnings that have not been refreshed for WARN_TTL_SECONDS
  (stale objects whose data was never updated are automatically cleared).
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from app.config import settings

logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────────
WARN_THRESHOLD_KM:  float = settings.CONJUNCTION_THRESHOLD_KM   # default 1.0 km
WARN_TTL_SECONDS:   float = 300.0   # expire a warning after 5 min without refresh
SCREEN_DEBRIS:      bool  = settings.DEBRIS_TRACKING_ENABLED


# ═══════════════════════════════════════════════════════════════════════════
# Warning record
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CDMWarning:
    """A live conjunction warning between two objects."""

    pair_key:           str       # canonical sorted pair, e.g. "id-A::id-B"
    object_a_id:        str
    object_b_id:        str
    object_a_type:      str       # "SATELLITE" | "DEBRIS"
    object_b_type:      str
    miss_distance_km:   float
    relative_speed_km_s: float
    risk_level:         str       # low | medium | high | critical
    first_detected_at:  float     # epoch seconds (time.monotonic)
    last_updated_at:    float     # epoch seconds – refreshed each screen pass

    def is_expired(self) -> bool:
        return (time.monotonic() - self.last_updated_at) > WARN_TTL_SECONDS

    def to_dict(self) -> dict:
        return {
            "pair_key":           self.pair_key,
            "object_a_id":        self.object_a_id,
            "object_b_id":        self.object_b_id,
            "object_a_type":      self.object_a_type,
            "object_b_type":      self.object_b_type,
            "miss_distance_km":   round(self.miss_distance_km, 4),
            "relative_speed_km_s": round(self.relative_speed_km_s, 4),
            "risk_level":         self.risk_level,
        }


def _risk(dist_km: float) -> str:
    if dist_km > 10.0:
        return "low"
    if dist_km > 2.0:
        return "medium"
    if dist_km > 0.5:
        return "high"
    return "critical"


# ═══════════════════════════════════════════════════════════════════════════
# Module-level warning store
# ═══════════════════════════════════════════════════════════════════════════

_warnings: Dict[str, CDMWarning] = {}   # keyed by pair_key


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

def active_warning_count() -> int:
    """Return the number of non-expired CDM warnings currently in the store."""
    _purge_expired()
    return len(_warnings)


def active_warnings() -> List[CDMWarning]:
    """Return all non-expired CDM warnings, sorted by miss distance."""
    _purge_expired()
    return sorted(_warnings.values(), key=lambda w: w.miss_distance_km)


def screen_updated_objects(updated_ids: List[Tuple[str, str]]) -> int:
    """
    Screen every object that appears in *updated_ids* against ALL other
    objects in the registry.

    Parameters
    ----------
    updated_ids : list[(object_id, object_type)]
        Objects whose state was just refreshed by the telemetry frame.

    Returns
    -------
    int
        Total number of active warnings (after this screening run).
    """
    from app.data.registry import satellites, debris   # lazy import avoids circular

    if not updated_ids:
        return active_warning_count()

    # Build flat snapshot:  {id: (type, x, y, z, vx, vy, vz)}
    snapshot: Dict[str, Tuple[str, float, float, float, float, float, float]] = {}

    for sat in satellites.values():
        snapshot[sat.id] = (
            "SATELLITE",
            sat.position.x, sat.position.y, sat.position.z,
            sat.velocity.vx, sat.velocity.vy, sat.velocity.vz,
        )
    if SCREEN_DEBRIS:
        for deb in debris.values():
            snapshot[deb.id] = (
                "DEBRIS",
                deb.position.x, deb.position.y, deb.position.z,
                deb.velocity.vx, deb.velocity.vy, deb.velocity.vz,
            )

    all_ids = list(snapshot.keys())
    updated_set = {oid for oid, _ in updated_ids}

    now = time.monotonic()
    checked = 0

    for i, id_a in enumerate(all_ids):
        # Only check pairs where at least one object was in the telemetry frame
        for id_b in all_ids[i + 1:]:
            if id_a not in updated_set and id_b not in updated_set:
                continue

            da = snapshot[id_a]
            db = snapshot[id_b]

            # Euclidean range
            dist = math.sqrt(
                (da[1] - db[1]) ** 2 +
                (da[2] - db[2]) ** 2 +
                (da[3] - db[3]) ** 2
            )
            checked += 1

            pair_key = f"{min(id_a, id_b)}::{max(id_a, id_b)}"

            if dist <= WARN_THRESHOLD_KM:
                rel_spd = math.sqrt(
                    (da[4] - db[4]) ** 2 +
                    (da[5] - db[5]) ** 2 +
                    (da[6] - db[6]) ** 2
                )
                risk = _risk(dist)
                if pair_key in _warnings:
                    warn = _warnings[pair_key]
                    warn.miss_distance_km    = dist
                    warn.relative_speed_km_s = rel_spd
                    warn.risk_level          = risk
                    warn.last_updated_at     = now
                else:
                    _warnings[pair_key] = CDMWarning(
                        pair_key           = pair_key,
                        object_a_id        = id_a,
                        object_b_id        = id_b,
                        object_a_type      = da[0],
                        object_b_type      = db[0],
                        miss_distance_km   = dist,
                        relative_speed_km_s = rel_spd,
                        risk_level         = risk,
                        first_detected_at  = now,
                        last_updated_at    = now,
                    )
                    logger.warning(
                        "CDM WARNING  %s ↔ %s  dist=%.3f km  risk=%s",
                        id_a, id_b, dist, risk,
                    )
            else:
                # Clear any previous warning for this pair
                _warnings.pop(pair_key, None)

    logger.debug("CDM screen: %d pairs checked, %d active warnings", checked, len(_warnings))
    _purge_expired()
    return len(_warnings)


def clear_warnings() -> None:
    """Clear all CDM warnings (used on registry reset)."""
    _warnings.clear()


# ═══════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════

def _purge_expired() -> None:
    expired = [k for k, w in _warnings.items() if w.is_expired()]
    for k in expired:
        del _warnings[k]
