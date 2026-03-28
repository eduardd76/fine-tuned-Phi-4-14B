"""
Virtual Architect Agent — Phi-4-14B Integration for Dream Team
Layer 3 specialist agent implementing the Dream Team A2A interface.

Capabilities:
  - network_topology_design
  - compliance_architecture (PCI-DSS, HIPAA, NIST CSF)
  - ha_design (99.99%+ uptime)
  - cost_estimation
  - technology_selection
  - implementation_planning

Communication:
  - Receives tasks from Team Leader via A2A (Redis queues)
  - Responds with reasoning chains (<think> tags) + final recommendations
  - Supports MCP protocol for tool use
  - Human-in-the-loop gate for high-risk design decisions
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("virtual_architect")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [VA] %(levelname)s %(message)s",
)

ROOT = Path(__file__).parent.parent

import sys
sys.path.insert(0, str(ROOT / "agents"))

# ─────────────────────────────────────────────────────────────────────────────
# Lazy imports for A2A/shared (only available in Docker environment)
# ─────────────────────────────────────────────────────────────────────────────

def _try_import_shared() -> tuple[Any, Any, Any]:
    try:
        from shared.a2a_protocol import A2AChannel, AgentRole  # type: ignore[import]
        from shared.mcp_client import MCPClient  # type: ignore[import]
        return A2AChannel, AgentRole, MCPClient
    except ImportError:
        logger.warning("shared/ not found — running in standalone mode")
        return None, None, None


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

MODEL_PATH = os.getenv(
    "VIRTUAL_ARCHITECT_MODEL",
    str(ROOT / "models" / "phi4-network-architect"),
)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
MCP_URL = os.getenv("MCP_URL", "http://mcp-server:8000")
MCP_API_KEY = os.getenv("MCP_API_KEY", "dev-key-123")
AGENT_INBOX = "a2a_virtual_architect_inbox"
CONFIDENCE_THRESHOLD = float(os.getenv("VA_CONFIDENCE_THRESHOLD", "0.80"))

CAPABILITIES = [
    "network_topology_design",
    "compliance_architecture",
    "ha_design",
    "cost_estimation",
    "technology_selection",
    "implementation_planning",
    "vxlan_bgp_evpn_design",
    "sd_wan_design",
    "mpls_vpn_design",
    "qos_design",
]

TASK_SYSTEM_PROMPTS: dict[str, str] = {
    "design": """You are a CCDE-level network architect. Design a complete network architecture
meeting all specified requirements. Use <think> tags to reason through topology selection,
redundancy, compliance, and cost. Provide Cisco IOS configuration examples where relevant.
Cite sources: [CCNP Enterprise Design], [Building Data Centers with VXLAN BGP EVPN], etc.""",

    "troubleshoot": """You are a CCDE network expert performing root cause analysis.
Use <think> tags to systematically diagnose the issue following OSI model bottom-up approach.
Provide step-by-step remediation with exact CLI commands.""",

    "estimate": """You are a network infrastructure cost analyst.
Use <think> tags to break down CAPEX and OPEX components.
Provide itemized estimates with market price ranges and optimization options.""",
}


# ─────────────────────────────────────────────────────────────────────────────
# Virtual Architect Agent
# ─────────────────────────────────────────────────────────────────────────────

class VirtualArchitectAgent:
    """
    Phi-4-14B based network architecture agent.
    Compatible with Dream Team multi-agent system.
    """

    def __init__(
        self,
        model_path: str = MODEL_PATH,
        config: dict[str, Any] | None = None,
    ) -> None:
        self._model_path = model_path
        self._config = config or {}
        self._engine: Any = None  # Lazy-loaded Phi4InferenceEngine
        self._redis: Any = None
        self._channel: Any = None

    # ── Lazy model loading ────────────────────────────────────────────────────

    def _get_engine(self) -> Any:
        if self._engine is None:
            from dream_team_integration.phi4_inference import Phi4InferenceEngine  # type: ignore[import]
            logger.info(f"Loading Phi-4 model: {self._model_path}")
            self._engine = Phi4InferenceEngine(
                model_path=self._model_path,
                cache_enabled=True,
                max_new_tokens=self._config.get("max_new_tokens", 2048),
                temperature=self._config.get("temperature", 0.7),
            )
        return self._engine

    # ── Public interface (matches Dream Team agent contract) ──────────────────

    async def receive_task(self, task: dict[str, Any]) -> dict[str, Any]:
        """
        Receive task from Team Leader.

        Args:
            task: {
                "type": "design" | "troubleshoot" | "estimate",
                "context": {...},
                "priority": "high" | "medium" | "low",
                "requestor": "team_leader" | ...,
            }

        Returns:
            {
                "status": "success" | "error",
                "reasoning": "<think>...</think>",
                "recommendation": "...",
                "confidence": 0.0-1.0,
                "sources": [...],
                "estimated_time": "...",
                "cost_estimate": {...},
                "requires_human_approval": bool,
            }
        """
        task_id = task.get("task_id", str(uuid.uuid4()))
        task_type = task.get("type", "design")
        context = task.get("context", {})

        logger.info(f"Task {task_id}: type={task_type}, priority={task.get('priority', 'medium')}")

        try:
            prompt = self._build_prompt(task_type, context)
            system_prompt = TASK_SYSTEM_PROMPTS.get(task_type, TASK_SYSTEM_PROMPTS["design"])

            engine = self._get_engine()
            result = engine.infer(prompt, system_override=system_prompt)

            logger.info(
                f"Task {task_id} complete: "
                f"confidence={result.confidence:.2f}, "
                f"steps={result.reasoning_steps}, "
                f"sources={len(result.sources)}"
            )

            # Determine if human approval needed
            requires_approval = (
                result.confidence < CONFIDENCE_THRESHOLD
                or task.get("priority") == "high"
                or context.get("compliance", [])  # Any compliance requirement
            )

            return {
                "status": "success",
                "task_id": task_id,
                "reasoning": f"<think>{result.reasoning}</think>",
                "recommendation": result.answer,
                "confidence": result.confidence,
                "sources": result.sources,
                "has_think_block": result.has_think_block,
                "reasoning_steps": result.reasoning_steps,
                "latency_ms": result.latency_ms,
                "requires_human_approval": requires_approval,
                "cost_estimate": self._extract_cost_estimate(result.answer),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            logger.error(f"Task {task_id} failed: {e}", exc_info=True)
            return {
                "status": "error",
                "task_id": task_id,
                "error": str(e),
                "reasoning": "",
                "recommendation": "",
                "confidence": 0.0,
                "sources": [],
                "requires_human_approval": True,
            }

    async def collaborate(self, agent_id: str, question: str) -> dict[str, Any]:
        """Ask another agent for information (stub — integrates via A2A channel)."""
        logger.info(f"Collaboration request to {agent_id}: {question[:60]}...")
        return {
            "agent_id": agent_id,
            "question": question,
            "response": f"[Collaboration with {agent_id} pending — implement via A2A channel]",
        }

    def get_capabilities(self) -> list[str]:
        """Return list of agent capabilities."""
        return CAPABILITIES

    # ── A2A event loop ────────────────────────────────────────────────────────

    async def start_a2a(self) -> None:
        """Start A2A event loop — called by docker-compose service."""
        A2AChannel, AgentRole, MCPClient = _try_import_shared()

        if A2AChannel is None:
            logger.error("Cannot start A2A loop — shared/ not available")
            return

        import redis.asyncio as aioredis  # type: ignore[import]
        self._redis = await aioredis.from_url(REDIS_URL, decode_responses=True)
        self._channel = A2AChannel(self._redis, AgentRole.VIRTUAL_ARCHITECT)

        logger.info(f"Virtual Architect listening on: {AGENT_INBOX}")

        while True:
            try:
                message = await self._channel.receive(
                    inbox=AGENT_INBOX, timeout=30
                )
                if message is None:
                    continue

                task_id = message.get("task_id", str(uuid.uuid4()))
                payload = message.get("payload", {})

                # Convert A2A message to task format
                task: dict[str, Any] = {
                    "task_id": task_id,
                    "type": payload.get("task_type", "design"),
                    "context": payload.get("context", payload),
                    "priority": payload.get("priority", "medium"),
                    "requestor": message.get("from_agent", "team_leader"),
                }

                result = await self.receive_task(task)

                await self._channel.send_task_response(
                    to_agent=AgentRole.TEAM_LEADER,
                    task_id=task_id,
                    correlation_id=message.get("correlation_id", task_id),
                    result=result,
                    success=result["status"] == "success",
                )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"A2A loop error: {e}", exc_info=True)
                await asyncio.sleep(5)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_prompt(self, task_type: str, context: dict[str, Any]) -> str:
        """Build a natural language prompt from task context."""
        if task_type == "design":
            req = context.get("requirements", context)
            users = req.get("users", "unspecified")
            sites = req.get("sites", 1)
            compliance = req.get("compliance", [])
            uptime = req.get("uptime", 99.9)
            budget = req.get("budget", "flexible")

            prompt = f"""Design a complete enterprise network architecture for:
- Users/endpoints: {users}
- Sites/locations: {sites}
- Required compliance: {', '.join(compliance) if compliance else 'none specified'}
- Uptime requirement: {uptime}% ({self._uptime_to_downtime(uptime)})
- Budget: {budget}

Provide topology selection rationale, redundancy design, routing protocol selection,
security architecture, and implementation roadmap. Include configuration examples."""

            # Append any additional context fields
            for key, val in req.items():
                if key not in ("users", "sites", "compliance", "uptime", "budget"):
                    prompt += f"\nAdditional requirement — {key}: {val}"

            return prompt

        elif task_type == "troubleshoot":
            return (
                f"Troubleshoot this network issue:\n\n"
                f"Device: {context.get('device', 'unknown')}\n"
                f"Symptom: {context.get('symptom', context.get('message', 'No description provided'))}\n"
                f"Protocol: {context.get('protocol', 'unknown')}\n"
                f"Impact: {context.get('impact', 'unknown')}\n"
            )

        elif task_type == "estimate":
            return (
                f"Provide a detailed cost estimate for:\n\n"
                f"{json.dumps(context, indent=2)}\n\n"
                f"Break down CAPEX and annual OPEX. Include hardware, licensing, and implementation."
            )

        else:
            return str(context)

    def _uptime_to_downtime(self, uptime: float) -> str:
        downtime_minutes = (100 - uptime) / 100 * 525600
        if downtime_minutes < 60:
            return f"{downtime_minutes:.1f} min/year"
        return f"{downtime_minutes/60:.1f} hours/year"

    def _extract_cost_estimate(self, answer: str) -> dict[str, Any]:
        """Extract cost figures from answer text if present."""
        import re
        capex_match = re.search(r"CAPEX[:\s]+\$?([\d,]+(?:\.\d+)?)\s*([MK]?)", answer, re.IGNORECASE)
        opex_match = re.search(r"annual\s+OPEX[:\s]+\$?([\d,]+(?:\.\d+)?)\s*([MK]?)", answer, re.IGNORECASE)

        def parse_amount(m: re.Match | None) -> float | None:
            if not m:
                return None
            num = float(m.group(1).replace(",", ""))
            suffix = m.group(2).upper()
            if suffix == "M":
                num *= 1_000_000
            elif suffix == "K":
                num *= 1_000
            return num

        capex = parse_amount(capex_match)
        opex = parse_amount(opex_match)

        return {
            "capex": capex,
            "annual_opex": opex,
            "currency": "USD",
        } if (capex or opex) else {}


import json  # noqa: E402 (needed for _build_prompt)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

async def main() -> None:
    agent = VirtualArchitectAgent()
    await agent.start_a2a()


if __name__ == "__main__":
    asyncio.run(main())
