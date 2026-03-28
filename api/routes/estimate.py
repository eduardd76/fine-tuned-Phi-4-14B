"""POST /api/v1/estimate"""
from __future__ import annotations
from typing import Any
from fastapi import APIRouter, Request, HTTPException, status
from api.models import EstimateRequest, EstimateResponse, CostBreakdown

router = APIRouter(tags=["estimate"])
SYSTEM = "You are a network infrastructure cost analyst. Itemize CAPEX and OPEX inside <think> tags."

@router.post("/estimate", response_model=EstimateResponse, status_code=status.HTTP_200_OK)
async def estimate(body: EstimateRequest, request: Request) -> EstimateResponse:
    engine: Any = getattr(request.app.state, "engine", None)
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    req = body.requirements
    prompt = f"Cost estimate for {req.users or '?'} users, {req.sites} sites, compliance={req.compliance}"
    result = engine.infer(prompt, system_override=SYSTEM, max_new_tokens=body.max_tokens)
    return EstimateResponse(
        reasoning=f"<think>{result.reasoning}</think>",
        breakdown=result.answer,
        cost_estimate=CostBreakdown(currency=body.currency),
        sources=result.sources,
        confidence=result.confidence,
        latency_ms=result.latency_ms,
    )
