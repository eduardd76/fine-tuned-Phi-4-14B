"""
Phi-4 Network Architect API
FastAPI server exposing network design capabilities.

Endpoints:
  POST /api/v1/design         — Full network architecture design
  POST /api/v1/troubleshoot   — Network troubleshooting
  POST /api/v1/estimate       — Cost estimation
  GET  /api/v1/health         — Health check
  GET  /api/v1/model/info     — Model information

Run:
  uvicorn api.main:app --host 0.0.0.0 --port 8000
  uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload  # dev mode
"""

from __future__ import annotations

import os
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes import design, troubleshoot, estimate, health
from api.middleware.auth import APIKeyMiddleware
from api.middleware.rate_limit import RateLimitMiddleware

logger = logging.getLogger("phi4_api")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [API] %(levelname)s %(message)s",
)

# ─────────────────────────────────────────────────────────────────────────────
# App lifecycle
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup: pre-load model. Shutdown: cleanup."""
    model_path = os.getenv(
        "VIRTUAL_ARCHITECT_MODEL",
        str(Path(__file__).parent.parent / "models" / "phi4-network-architect"),
    )

    if os.path.exists(model_path):
        logger.info(f"Pre-loading Phi-4 model: {model_path}")
        from dream_team_integration.phi4_inference import Phi4InferenceEngine  # type: ignore[import]
        app.state.engine = Phi4InferenceEngine(model_path=model_path)
        logger.info("Model loaded and ready")
    else:
        logger.warning(f"Model not found at {model_path} — run training pipeline first")
        app.state.engine = None

    yield

    logger.info("API server shutting down")


# ─────────────────────────────────────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Phi-4 Network Architect API",
    version="1.0.0",
    description="CCDE-level network architecture design powered by fine-tuned Phi-4-14B",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting (skip for health checks)
app.add_middleware(
    RateLimitMiddleware,
    requests_per_minute=int(os.getenv("RATE_LIMIT_RPM", "30")),
    exclude_paths=["/api/v1/health", "/docs", "/redoc", "/openapi.json"],
)

# API key auth (skip for health + docs)
app.add_middleware(
    APIKeyMiddleware,
    api_key_env="PHI4_API_KEY",
    exclude_paths=["/api/v1/health", "/docs", "/redoc", "/openapi.json"],
)

# ─────────────────────────────────────────────────────────────────────────────
# Routers
# ─────────────────────────────────────────────────────────────────────────────

app.include_router(health.router, prefix="/api/v1")
app.include_router(design.router, prefix="/api/v1")
app.include_router(troubleshoot.router, prefix="/api/v1")
app.include_router(estimate.router, prefix="/api/v1")


# ─────────────────────────────────────────────────────────────────────────────
# Error handlers
# ─────────────────────────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Any, exc: Exception) -> JSONResponse:
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PHI4_API_PORT", "8000")),
        reload=False,
    )
