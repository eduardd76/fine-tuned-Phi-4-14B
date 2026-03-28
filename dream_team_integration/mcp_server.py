"""
Dream Team MCP Server for Virtual Architect
Exposes Phi-4 network design capabilities as MCP tools.

Endpoints:
  POST /tools/network_design      — Full architecture design
  POST /tools/troubleshoot_design — Design-level troubleshooting
  POST /tools/cost_estimate       — Infrastructure cost estimation
  GET  /health                    — Service health check
  GET  /tools                     — List available tools
"""

from __future__ import annotations

import os
import time
import uuid
import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logger = logging.getLogger("va_mcp_server")

MODEL_PATH = os.getenv(
    "VIRTUAL_ARCHITECT_MODEL",
    str(Path(__file__).parent.parent / "models" / "phi4-network-architect"),
)
MCP_API_KEY = os.getenv("MCP_API_KEY", "dev-key-123")
PORT = int(os.getenv("MCP_SERVER_PORT", "5555"))

app = FastAPI(
    title="Virtual Architect MCP Server",
    version="1.0.0",
    description="Phi-4-14B network architecture tools via MCP protocol",
)

# Shared engine instance (lazy-loaded)
_engine: Any = None
_idempotency_cache: dict[str, Any] = {}


def get_engine() -> Any:
    global _engine
    if _engine is None:
        from dream_team_integration.phi4_inference import Phi4InferenceEngine  # type: ignore[import]
        logger.info(f"Loading Phi-4 model: {MODEL_PATH}")
        _engine = Phi4InferenceEngine(
            model_path=MODEL_PATH,
            cache_enabled=True,
        )
    return _engine


def verify_api_key(request: Request) -> None:
    key = request.headers.get("X-MCP-API-Key", "")
    if key != MCP_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


def check_idempotency(key: str) -> Any | None:
    entry = _idempotency_cache.get(key)
    if entry and time.time() - entry["ts"] < 86400:
        logger.debug(f"Idempotency hit: {key}")
        return entry["result"]
    return None


def store_idempotency(key: str, result: Any) -> None:
    _idempotency_cache[key] = {"result": result, "ts": time.time()}
    # Cleanup old entries (keep last 1000)
    if len(_idempotency_cache) > 1000:
        oldest = sorted(_idempotency_cache.items(), key=lambda x: x[1]["ts"])[:100]
        for k, _ in oldest:
            del _idempotency_cache[k]


# ─────────────────────────────────────────────────────────────────────────────
# Request/response models
# ─────────────────────────────────────────────────────────────────────────────

class NetworkRequirements(BaseModel):
    users: int | None = None
    sites: int = 1
    compliance: list[str] = Field(default_factory=list)
    uptime: float = 99.9
    budget: str = "flexible"
    wan_type: str | None = None
    data_center: bool = False
    wireless: bool = True

    class Config:
        extra = "allow"


class DesignRequest(BaseModel):
    requirements: NetworkRequirements
    idempotency_key: str = Field(default_factory=lambda: str(uuid.uuid4()))
    max_tokens: int = 2048
    temperature: float = 0.7


class TroubleshootRequest(BaseModel):
    symptom: str
    device: str = "unknown"
    protocol: str | None = None
    logs: str | None = None
    idempotency_key: str = Field(default_factory=lambda: str(uuid.uuid4()))


class CostEstimateRequest(BaseModel):
    requirements: NetworkRequirements
    include_opex: bool = True
    currency: str = "USD"
    idempotency_key: str = Field(default_factory=lambda: str(uuid.uuid4()))


class ToolResponse(BaseModel):
    tool: str
    reasoning: str
    result: str
    sources: list[str]
    confidence: float
    latency_ms: float
    idempotency_key: str
    cached: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# Tool endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "virtual-architect-mcp",
        "model": MODEL_PATH,
        "model_loaded": _engine is not None,
        "cache_size": len(_idempotency_cache),
    }


@app.get("/tools")
async def list_tools() -> dict[str, Any]:
    return {
        "tools": [
            {
                "name": "network_design",
                "description": "Design complete network architecture from requirements",
                "endpoint": "/tools/network_design",
            },
            {
                "name": "troubleshoot_design",
                "description": "Analyze and fix network design issues",
                "endpoint": "/tools/troubleshoot_design",
            },
            {
                "name": "cost_estimate",
                "description": "Estimate CAPEX and OPEX for network infrastructure",
                "endpoint": "/tools/cost_estimate",
            },
        ]
    }


@app.post("/tools/network_design", response_model=ToolResponse)
async def network_design(
    request: DesignRequest,
    _: None = Depends(verify_api_key),
) -> ToolResponse:
    """Design complete network architecture."""
    cached = check_idempotency(request.idempotency_key)
    if cached:
        return ToolResponse(**cached, cached=True)

    engine = get_engine()
    req = request.requirements

    prompt = f"""Design a complete enterprise network architecture:
Users: {req.users or 'not specified'}
Sites: {req.sites}
Compliance: {', '.join(req.compliance) if req.compliance else 'none'}
Uptime SLA: {req.uptime}%
Budget: {req.budget}
Data center needed: {req.data_center}
Wireless coverage: {req.wireless}

Provide: topology selection, routing design, redundancy, security segmentation,
hardware recommendations, and phased implementation plan."""

    from dream_team_integration.phi4_inference import TASK_SYSTEM_PROMPTS  # type: ignore[import]
    result = engine.infer(
        prompt,
        system_override=TASK_SYSTEM_PROMPTS.get("design"),
        max_new_tokens=request.max_tokens,
        temperature=request.temperature,
    )

    response_data = {
        "tool": "network_design",
        "reasoning": f"<think>{result.reasoning}</think>",
        "result": result.answer,
        "sources": result.sources,
        "confidence": result.confidence,
        "latency_ms": result.latency_ms,
        "idempotency_key": request.idempotency_key,
    }
    store_idempotency(request.idempotency_key, response_data)
    return ToolResponse(**response_data)


@app.post("/tools/troubleshoot_design", response_model=ToolResponse)
async def troubleshoot_design(
    request: TroubleshootRequest,
    _: None = Depends(verify_api_key),
) -> ToolResponse:
    """Troubleshoot network design issues."""
    cached = check_idempotency(request.idempotency_key)
    if cached:
        return ToolResponse(**cached, cached=True)

    engine = get_engine()
    prompt = (
        f"Troubleshoot network design issue:\n"
        f"Device: {request.device}\n"
        f"Symptom: {request.symptom}\n"
        f"Protocol: {request.protocol or 'unknown'}\n"
    )
    if request.logs:
        prompt += f"\nRelevant logs:\n{request.logs[:500]}"

    from dream_team_integration.phi4_inference import TASK_SYSTEM_PROMPTS  # type: ignore[import]
    result = engine.infer(prompt, system_override=TASK_SYSTEM_PROMPTS.get("troubleshoot"))

    response_data = {
        "tool": "troubleshoot_design",
        "reasoning": f"<think>{result.reasoning}</think>",
        "result": result.answer,
        "sources": result.sources,
        "confidence": result.confidence,
        "latency_ms": result.latency_ms,
        "idempotency_key": request.idempotency_key,
    }
    store_idempotency(request.idempotency_key, response_data)
    return ToolResponse(**response_data)


@app.post("/tools/cost_estimate", response_model=ToolResponse)
async def cost_estimate(
    request: CostEstimateRequest,
    _: None = Depends(verify_api_key),
) -> ToolResponse:
    """Estimate infrastructure costs."""
    cached = check_idempotency(request.idempotency_key)
    if cached:
        return ToolResponse(**cached, cached=True)

    engine = get_engine()
    req = request.requirements
    prompt = (
        f"Provide detailed cost estimate ({request.currency}):\n"
        f"Users: {req.users or 'unspecified'}, Sites: {req.sites}\n"
        f"Compliance: {', '.join(req.compliance) if req.compliance else 'none'}\n"
        f"Include: CAPEX itemization, {'annual OPEX, ' if request.include_opex else ''}"
        f"hardware, licensing, implementation labor"
    )

    from dream_team_integration.phi4_inference import TASK_SYSTEM_PROMPTS  # type: ignore[import]
    result = engine.infer(prompt, system_override=TASK_SYSTEM_PROMPTS.get("estimate"))

    response_data = {
        "tool": "cost_estimate",
        "reasoning": f"<think>{result.reasoning}</think>",
        "result": result.answer,
        "sources": result.sources,
        "confidence": result.confidence,
        "latency_ms": result.latency_ms,
        "idempotency_key": request.idempotency_key,
    }
    store_idempotency(request.idempotency_key, response_data)
    return ToolResponse(**response_data)


@app.exception_handler(Exception)
async def global_error_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error(f"MCP server error: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": str(exc)})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
