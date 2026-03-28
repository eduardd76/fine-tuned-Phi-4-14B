"""GET /api/v1/health"""
from __future__ import annotations
import os
from typing import Any
from fastapi import APIRouter, Request
router = APIRouter(tags=["health"])

@router.get("/health")
async def health(request: Request) -> dict[str, Any]:
    engine: Any = getattr(request.app.state, "engine", None)
    return {
        "status": "ok",
        "model_loaded": engine is not None,
        "model_path": os.getenv("VIRTUAL_ARCHITECT_MODEL", "not set"),
        "service": "phi4-network-architect-api",
        "version": "1.0.0",
    }
