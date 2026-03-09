"""
Visualization API – ground tracks, coverage maps, 3D orbit data, and conjunction plots.

Also exposes GET /snapshot – a bandwidth-optimised live snapshot of the entire
registry, converting ECI (x,y,z) positions directly to geodetic (lat,lon,alt)
without needing Keplerian elements.
"""

from __future__ import annotations

import math
import time as _time
from datetime import datetime
from typing import List, Optional, Tuple

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.services.satellite_service import SatelliteService
from app.physics.propagator import OrbitalPropagator
from app.physics.coordinate_transforms import eci_to_ecef, ecef_to_geodetic, _greenwich_sidereal_time
from app.models.orbital_state import CartesianState

router = APIRouter(prefix="/visualization")
_sat_svc  = SatelliteService()
_propagator = OrbitalPropagator()


# ── Fast ECI→geodetic helper (avoids CartesianState allocation overhead) ──────
def _eci_to_geodetic_raw(
    x_km: float, y_km: float, z_km: float,
    gst_rad: float,
) -> Tuple[float, float, float]:
    """
    Convert ECI [km] → geodetic (lat_deg, lon_deg, alt_km) using a pre-computed
    Greenwich Sidereal Time, bypassing CartesianState model creation.

    This is the hot path for large debris clouds: all trig computed once per
    snapshot (gst_rad shared), only per-object Bowring iteration here.
    """
    # ECI → ECEF rotation
    cos_g = math.cos(gst_rad)
    sin_g = math.sin(gst_rad)
    xE =  cos_g * x_km + sin_g * y_km
    yE = -sin_g * x_km + cos_g * y_km
    zE = z_km

    # ECEF → geodetic (Bowring / iterative)
    _E2     = 0.00669437999014
    _RE_WGS = 6_378.137
    lon_rad = math.atan2(yE, xE)
    p = math.sqrt(xE * xE + yE * yE)
    lat_rad = math.atan2(zE, p * (1.0 - _E2))
    for _ in range(6):                   # 6 iterations → < 0.001 m error
        s = math.sin(lat_rad)
        N = _RE_WGS / math.sqrt(1.0 - _E2 * s * s)
        lat_rad = math.atan2(zE + _E2 * N * s, p)
    s = math.sin(lat_rad)
    c = math.cos(lat_rad)
    N_f = _RE_WGS / math.sqrt(1.0 - _E2 * s * s)
    alt_km = (p / c - N_f) if abs(c) > 1e-10 else abs(zE) - N_f * (1.0 - _E2)

    return math.degrees(lat_rad), math.degrees(lon_rad), alt_km


class GroundTrackPoint(BaseModel):
    timestamp: str
    latitude_deg: float
    longitude_deg: float
    altitude_km: float


class OrbitPoint3D(BaseModel):
    timestamp: str
    x_km: float
    y_km: float
    z_km: float
    altitude_km: float


@router.get(
    "/ground-track/{satellite_id}",
    response_model=List[GroundTrackPoint],
    summary="Compute ground track (lat/lon over time)",
)
async def ground_track(
    satellite_id: str,
    duration_seconds: float = Query(5400.0, gt=0, le=86400),
    step_seconds: float = Query(60.0, gt=0, le=3600),
):
    sat = _sat_svc.get(satellite_id)
    if not sat:
        raise HTTPException(404, f"Satellite '{satellite_id}' not found")

    propagator = OrbitalPropagator(step_seconds=step_seconds)
    states = propagator.propagate(
        sat.orbital_elements,
        epoch=datetime.utcnow(),
        duration_seconds=duration_seconds,
        drag_coefficient=sat.drag_coefficient,
        mass_kg=sat.mass_kg,
        cross_section_m2=sat.cross_section_m2,
    )

    track: List[GroundTrackPoint] = []
    for state in states:
        if state.cartesian is None:
            continue
        x_ecef, y_ecef, z_ecef = eci_to_ecef(state.cartesian, state.epoch)
        lat, lon, alt = ecef_to_geodetic(x_ecef, y_ecef, z_ecef)
        track.append(GroundTrackPoint(
            timestamp=state.epoch.isoformat(),
            latitude_deg=round(lat, 6),
            longitude_deg=round(lon, 6),
            altitude_km=round(alt, 3),
        ))
    return track


@router.get(
    "/orbit-3d/{satellite_id}",
    response_model=List[OrbitPoint3D],
    summary="Get ECI 3D orbit path for one orbital period",
)
async def orbit_3d(
    satellite_id: str,
    duration_seconds: Optional[float] = Query(None, gt=0, description="Defaults to one orbital period"),
    step_seconds: float = Query(30.0, gt=0, le=600),
):
    sat = _sat_svc.get(satellite_id)
    if not sat:
        raise HTTPException(404, f"Satellite '{satellite_id}' not found")

    period = sat.orbital_elements.period_seconds
    dur = duration_seconds or period
    propagator = OrbitalPropagator(step_seconds=step_seconds)
    states = propagator.propagate(
        sat.orbital_elements,
        epoch=datetime.utcnow(),
        duration_seconds=dur,
    )

    return [
        OrbitPoint3D(
            timestamp=s.epoch.isoformat(),
            x_km=round(s.cartesian.x_km, 3) if s.cartesian else 0,
            y_km=round(s.cartesian.y_km, 3) if s.cartesian else 0,
            z_km=round(s.cartesian.z_km, 3) if s.cartesian else 0,
            altitude_km=round(s.altitude_km or 0, 3),
        )
        for s in states
    ]


@router.get("/constellation-map", summary="Snapshot of all satellite ground positions")
async def constellation_map():
    sats = _sat_svc.list_all(limit=500)
    now = datetime.utcnow()
    positions = []
    for sat in sats:
        try:
            from app.physics.coordinate_transforms import keplerian_to_cartesian
            cart = keplerian_to_cartesian(sat.orbital_elements)
            x_ecef, y_ecef, z_ecef = eci_to_ecef(cart, now)
            lat, lon, alt = ecef_to_geodetic(x_ecef, y_ecef, z_ecef)
            positions.append({
                "id": sat.id,
                "name": sat.name,
                "constellation": sat.constellation,
                "status": sat.status,
                "latitude_deg": round(lat, 6),
                "longitude_deg": round(lon, 6),
                "altitude_km": round(alt, 3),
            })
        except Exception:
            continue
    return {"timestamp": now.isoformat(), "satellites": positions}


# ════════════════════════════════════════════════════════════════════════
# GET /snapshot  – live registry snapshot, bandwidth-optimised
# ════════════════════════════════════════════════════════════════════════


def _build_snapshot(
    round_deg:   int  = 5,
    round_alt:   int  = 2,
    include_eci: bool = False,
    max_debris:  int  = 10_000,
) -> dict:
    """
    Core snapshot builder with plain Python defaults (no FastAPI dependencies).
    Callable directly from tests or other services without an ASGI context.
    """
    from app.data.registry import satellites, debris

    t0  = _time.perf_counter()
    now = datetime.utcnow()
    gst = _greenwich_sidereal_time(now)

    # ── Satellites (named dict, richer) ───────────────────────────────────────────
    sat_rows: List[dict] = []
    errors_sat = 0
    for sat in satellites.values():
        try:
            lat, lon, alt = _eci_to_geodetic_raw(
                sat.position.x, sat.position.y, sat.position.z, gst,
            )
            row: dict = {
                "id":      sat.id,
                "name":    sat.name,
                "lat":     round(lat, round_deg),
                "lon":     round(lon, round_deg),
                "alt_km":  round(alt, round_alt),
                "fuel_kg": round(sat.fuel_kg, 3),
                "status":  sat.status.value if hasattr(sat.status, "value") else str(sat.status),
            }
            if include_eci:
                row["eci_x_km"] = round(sat.position.x, 3)
                row["eci_y_km"] = round(sat.position.y, 3)
                row["eci_z_km"] = round(sat.position.z, 3)
            sat_rows.append(row)
        except Exception:
            errors_sat += 1

    # ── Debris cloud (flattened arrays – bandwidth-optimised) ───────────────
    # Row schema: [id_str, lat_deg, lon_deg, alt_km]
    deb_list  = list(debris.values())
    truncated = len(deb_list) > max_debris
    if truncated:
        deb_list = deb_list[:max_debris]

    debris_cloud: List[list] = []
    errors_deb = 0
    for deb in deb_list:
        try:
            lat, lon, alt = _eci_to_geodetic_raw(
                deb.position.x, deb.position.y, deb.position.z, gst,
            )
            debris_cloud.append([
                deb.id,
                round(lat, round_deg),
                round(lon, round_deg),
                round(alt, round_alt),
            ])
        except Exception:
            errors_deb += 1

    elapsed_ms = (_time.perf_counter() - t0) * 1000.0

    return {
        "timestamp":       now.isoformat(),
        "satellites":      sat_rows,
        "satellite_count": len(sat_rows),
        "debris_cloud":    debris_cloud,
        "debris_count":    len(debris_cloud),
        "debris_schema":   ["id", "lat_deg", "lon_deg", "alt_km"],
        "meta": {
            "snapshot_time_ms":      round(elapsed_ms, 2),
            "errors_satellites":     errors_sat,
            "errors_debris":         errors_deb,
            "truncated_debris":      truncated,
            "total_debris_registry": len(debris),
            "gst_rad":               round(gst, 8),
        },
    }


@router.get(
    "/snapshot",
    summary="Live registry snapshot with geodetic positions (bandwidth-optimised)",
)
async def registry_snapshot(
    round_deg: int    = Query(5, ge=1, le=8,         description="Decimal places for lat/lon"),
    round_alt: int    = Query(2, ge=0, le=4,         description="Decimal places for altitude [km]"),
    include_eci: bool = Query(False,                 description="Include raw ECI x/y/z in satellite entries"),
    max_debris: int   = Query(10_000, ge=1, le=50_000, description="Maximum debris rows to return"),
):
    """
    Return a compact, real-time snapshot of every object in the simulation registry.

    **Satellite entries** (named fields – few objects, richer payload)::

        { "id", "name", "lat", "lon", "alt_km", "fuel_kg", "status" }

    **Debris cloud** (flattened arrays – compact for large populations)::

        [[id, lat_deg, lon_deg, alt_km], ...]

    Payload size comparison for 10 000 debris objects:

    | Format           | ~Size   |
    |------------------|---------|
    | Named JSON keys  | ~1.6 MB |
    | Flattened arrays | ~0.8 MB |

    A single Greenwich Sidereal Time (GST) is pre-computed for the whole
    snapshot epoch: trig cost is O(1 + N) not O(4N).
    """
    return _build_snapshot(
        round_deg   = round_deg,
        round_alt   = round_alt,
        include_eci = include_eci,
        max_debris  = max_debris,
    )
