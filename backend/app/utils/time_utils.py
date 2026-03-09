"""
Time utilities.
"""
from __future__ import annotations
from datetime import datetime, timezone


def utcnow_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def parse_iso(s: str) -> datetime:
    """Parse ISO 8601 string to datetime (UTC-aware)."""
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def julian_date(dt: datetime) -> float:
    """Convert datetime to Julian Date."""
    year, month, day = dt.year, dt.month, dt.day
    hour = dt.hour + dt.minute / 60.0 + dt.second / 3600.0 + dt.microsecond / 3_600_000_000.0
    if month <= 2:
        year -= 1
        month += 12
    A = int(year / 100)
    B = 2 - A + int(A / 4)
    JD = int(365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + day + hour / 24.0 + B - 1524.5
    return JD


def days_since_j2000(dt: datetime) -> float:
    j2000 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return (dt - j2000).total_seconds() / 86400.0
