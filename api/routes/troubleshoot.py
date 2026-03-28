"""POST /api/v1/troubleshoot"""
from __future__ import annotations
import re, logging
from typing import Any
from fastapi import APIRouter, Request, HTTPException, status
from api.models import TroubleshootRequest, TroubleshootResponse

logger = logging.getLogger("phi4_api.troubleshoot")
router = APIRouter(tags=["troubleshoot"])

SYSTEM = """You are a CCDE network expert. Diagnose the issue using OSI bottom-up methodology
inside <think> tags. Provide numbered remediation steps with exact Cisco CLI commands.
Mark destructive commands with [CAUTION]. Cite RFCs and design guides."""

@router.post("/troubleshoot", response_model=TroubleshootResponse, status_code=status.HTTP_200_OK)
async def troubleshoot(body: TroubleshootRequest, request: Request) -> TroubleshootResponse:
    engine: Any = getattr(request.app.state, "engine", None)
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    prompt = f"Troubleshoot: {body.symptom}\nDevice: {body.device}\nProtocol: {body.protocol or 'unknown'}"
    if body.logs:
        prompt += f"\nLogs:\n{body.logs[:500]}"

    result = engine.infer(prompt, system_override=SYSTEM, max_new_tokens=body.max_tokens)
    steps = re.findall(r"Step\s+\d+[^\n]*(?:\n(?!Step\s+\d)[^\n]*)*", result.answer)

    return TroubleshootResponse(
        reasoning=f"<think>{result.reasoning}</think>",
        diagnosis=result.answer,
        remediation_steps=steps or [result.answer],
        sources=result.sources,
        confidence=result.confidence,
        latency_ms=result.latency_ms,
    )
