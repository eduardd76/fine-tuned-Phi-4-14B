"""POST /api/v1/design — Network architecture design endpoint."""

from __future__ import annotations

import re
import logging
from typing import Any

from fastapi import APIRouter, Request, HTTPException, status

from api.models import DesignRequest, DesignResponse, CostBreakdown

logger = logging.getLogger("phi4_api.design")
router = APIRouter(tags=["design"])

DESIGN_SYSTEM = """You are a CCDE-level network architect. Design a complete network architecture
meeting all specified requirements. Reason step-by-step inside <think> tags, then provide
your final recommendation. Include:
1. Topology selection with justification (three-tier/spine-leaf/collapsed core)
2. Routing protocol design (OSPF/BGP/IS-IS selection)
3. Redundancy and HA design
4. Security segmentation (VLANs, VRFs, firewall placement)
5. Hardware recommendations with model numbers
6. Compliance-specific controls
7. Phased implementation roadmap

Cite sources: [CCNP Enterprise Design ENSLD], [Building Data Centers with VXLAN BGP EVPN],
[PCI-DSS 4.0.1], [HIPAA 2013], [RFC numbers], etc."""


def _parse_cost(answer: str) -> CostBreakdown:
    """Extract cost figures from design text."""
    def extract_usd(pattern: str) -> float | None:
        m = re.search(pattern, answer, re.IGNORECASE)
        if not m:
            return None
        raw = m.group(1).replace(",", "").replace("$", "")
        multiplier = {"M": 1_000_000, "K": 1_000}.get(m.group(2, ).upper() if m.lastindex and m.lastindex >= 2 else "", 1)
        try:
            return float(raw) * multiplier
        except ValueError:
            return None

    capex_m = re.search(r"CAPEX[:\s\$]*([0-9,]+(?:\.[0-9]+)?)\s*([MKmk]?)", answer)
    opex_m = re.search(r"(?:annual\s+)?OPEX[:\s\$]*([0-9,]+(?:\.[0-9]+)?)\s*([MKmk]?)", answer)

    def parse_val(m: re.Match | None) -> float | None:
        if not m:
            return None
        num = float(m.group(1).replace(",", ""))
        suffix = (m.group(2) or "").upper()
        return num * {"M": 1_000_000, "K": 1_000}.get(suffix, 1)

    return CostBreakdown(
        capex=parse_val(capex_m),
        annual_opex=parse_val(opex_m),
        currency="USD",
    )


def _requires_review(confidence: float, compliance: list[str], uptime: float) -> bool:
    return (
        confidence < 0.80
        or bool(compliance)
        or uptime >= 99.99
    )


@router.post(
    "/design",
    response_model=DesignResponse,
    status_code=status.HTTP_200_OK,
    summary="Design network architecture",
)
async def design_network(
    request_body: DesignRequest,
    request: Request,
) -> DesignResponse:
    """
    Generate a complete network architecture design from requirements.

    Returns reasoning chain, final design, cost estimate, and source citations.
    """
    engine: Any = getattr(request.app.state, "engine", None)
    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Run training pipeline first.",
        )

    req = request_body.requirements
    prompt = (
        f"Design a complete enterprise network architecture:\n"
        f"- Users/endpoints: {req.users or 'not specified'}\n"
        f"- Sites: {req.sites}\n"
        f"- Compliance: {', '.join(req.compliance) if req.compliance else 'none'}\n"
        f"- Uptime SLA: {req.uptime}%\n"
        f"- Budget: {req.budget}\n"
        f"- Data center: {req.data_center}\n"
        f"- WAN type: {req.wan_type or 'not specified'}\n"
        f"- Cloud connectivity: {req.cloud_connectivity}\n"
        f"- Redundancy level: {req.redundancy_level}\n"
    )

    # Append extra fields
    extra = req.model_extra or {}
    for k, v in extra.items():
        prompt += f"- {k}: {v}\n"

    logger.info(f"Design request: users={req.users}, sites={req.sites}, compliance={req.compliance}")

    result = engine.infer(
        prompt,
        system_override=DESIGN_SYSTEM,
        max_new_tokens=request_body.max_tokens,
        temperature=request_body.temperature,
        use_cache=request_body.use_cache,
    )

    return DesignResponse(
        reasoning=f"<think>{result.reasoning}</think>" if result.reasoning else "",
        design=result.answer,
        cost_estimate=_parse_cost(result.answer),
        sources=result.sources,
        confidence=result.confidence,
        latency_ms=result.latency_ms,
        requires_human_review=_requires_review(
            result.confidence, req.compliance, req.uptime
        ),
    )
