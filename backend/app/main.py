"""
Autonomous Constellation Manager - FastAPI Application Entry Point
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from contextlib import asynccontextmanager
import logging

from app.config import settings
from app.api import (satellites, debris, telemetry, maneuvers, simulation,
                     visualization, health, registry, propagation, collision,
                     cdm, avoidance)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan – startup / shutdown hooks
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Autonomous Constellation Manager starting up …")
    logger.info(f"   Environment : {settings.ENVIRONMENT}")
    logger.info(f"   Simulation  : {settings.SIMULATION_STEP_SECONDS}s step | {settings.MAX_PROPAGATION_DAYS}d max propagation")
    yield
    logger.info("🛑 Autonomous Constellation Manager shutting down …")


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------
def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.PROJECT_NAME,
        description=(
            "REST API for simulating and managing satellite constellations, "
            "orbital mechanics, debris tracking, maneuver planning, and telemetry."
        ),
        version=settings.VERSION,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # ── Middleware ──────────────────────────────────────────────────────────
    app.add_middleware(GZipMiddleware, minimum_size=1024)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routers ─────────────────────────────────────────────────────────────
    prefix = settings.API_V1_PREFIX
    app.include_router(health.router,         prefix=prefix, tags=["Health"])
    app.include_router(satellites.router,     prefix=prefix, tags=["Satellites"])
    app.include_router(debris.router,         prefix=prefix, tags=["Debris"])
    app.include_router(telemetry.router,      prefix=prefix, tags=["Telemetry"])
    app.include_router(maneuvers.router,      prefix=prefix, tags=["Maneuvers"])
    app.include_router(simulation.router,     prefix=prefix, tags=["Simulation"])
    app.include_router(visualization.router,  prefix=prefix, tags=["Visualization"])
    app.include_router(registry.router,       prefix=prefix, tags=["Registry"])
    app.include_router(propagation.router,    prefix=prefix, tags=["Propagation"])
    app.include_router(collision.router,      prefix=prefix, tags=["Collision"])
    app.include_router(cdm.router,            prefix=prefix, tags=["CDM"])
    app.include_router(avoidance.router,      prefix=prefix, tags=["Avoidance"])

    return app


app = create_app()
