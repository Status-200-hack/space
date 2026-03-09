"""
Unit conversion helpers.
"""
import math


def degrees_to_radians(deg: float) -> float:
    return math.radians(deg)


def radians_to_degrees(rad: float) -> float:
    return math.degrees(rad)


def km_to_meters(km: float) -> float:
    return km * 1000.0


def meters_to_km(m: float) -> float:
    return m / 1000.0


def km_s_to_m_s(km_s: float) -> float:
    return km_s * 1000.0


def m_s_to_km_s(m_s: float) -> float:
    return m_s / 1000.0
