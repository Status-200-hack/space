"""
Coordinate transformation utilities.

Converts between:
  • Keplerian ↔ Cartesian (ECI)
  • ECI → ECEF (Earth-Centred Earth-Fixed)
  • ECEF → Geodetic (lat/lon/alt)
"""

from __future__ import annotations

import math
from datetime import datetime

from app.models.orbital_state import KeplerianElements, CartesianState


# Earth parameters
_MU = 398_600.4418      # km³/s²
_RE = 6_371.0           # km
_E2 = 0.00669437999014  # WGS84 first eccentricity squared
_RE_WGS = 6_378.137     # km (WGS84 equatorial)
_OMEGA_EARTH = 7.2921150e-5  # rad/s


def keplerian_to_cartesian(kep: KeplerianElements) -> CartesianState:
    """Convert Keplerian elements to ECI Cartesian state vector."""
    a = kep.semi_major_axis_km
    e = kep.eccentricity
    i = math.radians(kep.inclination_deg)
    raan = math.radians(kep.raan_deg)
    omega = math.radians(kep.arg_of_perigee_deg)
    nu = math.radians(kep.true_anomaly_deg)

    # Perifocal (PQW) frame
    p = a * (1 - e ** 2)
    cos_nu = math.cos(nu)
    sin_nu = math.sin(nu)
    denom = 1 + e * cos_nu

    r_pqw = (p / denom) * _arr(cos_nu, sin_nu, 0)
    v_pqw = math.sqrt(_MU / p) * _arr(-sin_nu, e + cos_nu, 0)

    # Rotation matrix PQW → ECI
    R = _rot_pqw_to_eci(i, raan, omega)

    r_eci = _mat_vec(R, r_pqw)
    v_eci = _mat_vec(R, v_pqw)

    return CartesianState(
        x_km=r_eci[0], y_km=r_eci[1], z_km=r_eci[2],
        vx_km_s=v_eci[0], vy_km_s=v_eci[1], vz_km_s=v_eci[2],
    )


def cartesian_to_keplerian(cart: CartesianState) -> KeplerianElements:
    """Convert ECI Cartesian state to Keplerian elements."""
    r = [cart.x_km, cart.y_km, cart.z_km]
    v = [cart.vx_km_s, cart.vy_km_s, cart.vz_km_s]

    r_norm = _norm(r)
    v_norm = _norm(v)

    # Specific angular momentum
    h = _cross(r, v)
    h_norm = _norm(h)

    # Node vector
    K = [0, 0, 1]
    N = _cross(K, h)
    N_norm = _norm(N)

    # Eccentricity vector
    e_vec = [
        (v_norm ** 2 / _MU - 1 / r_norm) * r[j] - (_dot(r, v) / _MU) * v[j]
        for j in range(3)
    ]
    e = _norm(e_vec)

    # Semi-major axis
    energy = v_norm ** 2 / 2 - _MU / r_norm
    a = -_MU / (2 * energy)

    # Inclination
    i_rad = math.acos(max(-1, min(1, h[2] / h_norm)))

    # RAAN
    raan = math.atan2(N[1], N[0]) % (2 * math.pi) if N_norm > 1e-10 else 0.0

    # Argument of perigee
    if N_norm > 1e-10:
        cos_omega = _dot(N, e_vec) / (N_norm * e)
        omega = math.acos(max(-1, min(1, cos_omega)))
        if e_vec[2] < 0:
            omega = 2 * math.pi - omega
    else:
        omega = 0.0

    # True anomaly
    cos_nu = _dot(e_vec, r) / (e * r_norm) if e > 1e-10 else _dot(N, r) / (N_norm * r_norm)
    nu = math.acos(max(-1, min(1, cos_nu)))
    if _dot(r, v) < 0:
        nu = 2 * math.pi - nu

    return KeplerianElements(
        semi_major_axis_km=a,
        eccentricity=max(0.0, min(0.9999, e)),
        inclination_deg=math.degrees(i_rad),
        raan_deg=math.degrees(raan),
        arg_of_perigee_deg=math.degrees(omega),
        true_anomaly_deg=math.degrees(nu % (2 * math.pi)),
    )


def eci_to_ecef(cart: CartesianState, epoch: datetime) -> tuple[float, float, float]:
    """Rotate ECI to ECEF using Greenwich Sidereal Time."""
    gst = _greenwich_sidereal_time(epoch)
    cos_g = math.cos(gst)
    sin_g = math.sin(gst)
    x_ecef = cos_g * cart.x_km + sin_g * cart.y_km
    y_ecef = -sin_g * cart.x_km + cos_g * cart.y_km
    z_ecef = cart.z_km
    return x_ecef, y_ecef, z_ecef


def ecef_to_geodetic(x_km: float, y_km: float, z_km: float) -> tuple[float, float, float]:
    """Convert ECEF to geodetic (lat_deg, lon_deg, alt_km) using Bowring's method."""
    lon_rad = math.atan2(y_km, x_km)
    p = math.sqrt(x_km ** 2 + y_km ** 2)
    lat_rad = math.atan2(z_km, p * (1 - _E2))
    for _ in range(10):
        sin_lat = math.sin(lat_rad)
        N = _RE_WGS / math.sqrt(1 - _E2 * sin_lat ** 2)
        lat_rad = math.atan2(z_km + _E2 * N * sin_lat, p)
    sin_lat = math.sin(lat_rad)
    N_final = _RE_WGS / math.sqrt(1 - _E2 * sin_lat ** 2)
    alt_km = p / math.cos(lat_rad) - N_final if abs(math.cos(lat_rad)) > 1e-10 else abs(z_km) - N_final * (1 - _E2)
    return math.degrees(lat_rad), math.degrees(lon_rad), alt_km


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _arr(*args):
    return list(args)

def _norm(v):
    return math.sqrt(sum(x ** 2 for x in v))

def _dot(a, b):
    return sum(ai * bi for ai, bi in zip(a, b))

def _cross(a, b):
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]

def _rot_pqw_to_eci(i, raan, omega):
    cos_r, sin_r = math.cos(raan), math.sin(raan)
    cos_i, sin_i = math.cos(i), math.sin(i)
    cos_o, sin_o = math.cos(omega), math.sin(omega)
    return [
        [cos_r * cos_o - sin_r * sin_o * cos_i,  -cos_r * sin_o - sin_r * cos_o * cos_i,  sin_r * sin_i],
        [sin_r * cos_o + cos_r * sin_o * cos_i,  -sin_r * sin_o + cos_r * cos_o * cos_i, -cos_r * sin_i],
        [sin_o * sin_i,                            cos_o * sin_i,                           cos_i],
    ]

def _mat_vec(M, v):
    return [sum(M[i][j] * v[j] for j in range(3)) for i in range(3)]

def _greenwich_sidereal_time(epoch: datetime) -> float:
    """Approximate GST in radians."""
    j2000 = datetime(2000, 1, 1, 12, 0, 0)
    T = (epoch - j2000).total_seconds() / 86400.0 / 36525.0
    gst_deg = 280.46061837 + 360.98564736629 * (epoch - j2000).total_seconds() / 86400.0 + 0.000387933 * T ** 2
    return math.radians(gst_deg % 360)
