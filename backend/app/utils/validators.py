"""
General-purpose validators.
"""
from __future__ import annotations
import re


_TLE_LINE1_RE = re.compile(r"^1 \d{5}[A-Z] .{6} \d{2}\d{3}\.\d{8} [+-]\.\d{8} [+-]\d{5}[+-]\d [+-]\d{5}[+-]\d \d \s*\d{4}\d$")
_TLE_LINE2_RE = re.compile(r"^2 \d{5} .*\d{4}$")


def validate_tle_pair(line1: str, line2: str) -> bool:
    """
    Basic TLE format check.
    Returns True if both lines appear structurally valid.
    """
    if not (line1 and line2):
        return False
    if not (line1.startswith("1 ") and line2.startswith("2 ")):
        return False
    if len(line1) != 69 or len(line2) != 69:
        return False
    return _tle_checksum(line1) and _tle_checksum(line2)


def _tle_checksum(line: str) -> bool:
    total = 0
    for ch in line[:-1]:
        if ch.isdigit():
            total += int(ch)
        elif ch == "-":
            total += 1
    return (total % 10) == int(line[-1])
