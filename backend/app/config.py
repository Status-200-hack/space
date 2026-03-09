"""
Application configuration – driven by environment variables via Pydantic Settings.
"""

from __future__ import annotations

from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Project metadata ────────────────────────────────────────────────────
    PROJECT_NAME: str = "Autonomous Constellation Manager"
    VERSION: str = "0.1.0"
    ENVIRONMENT: str = "development"          # development | staging | production

    # ── API ─────────────────────────────────────────────────────────────────
    API_V1_PREFIX: str = "/api/v1"
    PORT: int = 8000
    HOST: str = "0.0.0.0"

    # ── CORS ────────────────────────────────────────────────────────────────
    ALLOWED_ORIGINS: List[str] = ["*"]

    # ── Physics / Simulation ────────────────────────────────────────────────
    SIMULATION_STEP_SECONDS: float = 60.0          # propagation time step
    MAX_PROPAGATION_DAYS: float = 7.0              # maximum forward propagation
    EARTH_RADIUS_KM: float = 6371.0
    EARTH_MU_KM3_S2: float = 398_600.4418          # gravitational parameter μ
    J2_COEFFICIENT: float = 1.08263e-3             # J2 oblateness perturbation
    SPEED_OF_LIGHT_KM_S: float = 299_792.458

    # ── Collision / Conjunction ──────────────────────────────────────────────
    CONJUNCTION_THRESHOLD_KM: float = 1.0          # miss-distance alert threshold
    DEBRIS_TRACKING_ENABLED: bool = True

    # ── Telemetry ────────────────────────────────────────────────────────────
    TELEMETRY_BUFFER_SIZE: int = 10_000            # in-memory ring-buffer entries
    TELEMETRY_RETENTION_DAYS: int = 30

    # ── Maneuver Planning ────────────────────────────────────────────────────
    MAX_DELTA_V_M_S: float = 500.0                 # maximum allowable Δv per burn
    DEFAULT_ISP_SECONDS: float = 220.0             # specific impulse (cold gas)

    # ── Data / Storage ───────────────────────────────────────────────────────
    DATA_DIR: str = "app/data"
    TLE_CACHE_TTL_SECONDS: int = 3600


settings = Settings()
