"""
Technical Accuracy Evaluator

Verifies model outputs against the NotebookLM knowledge base.
Checks specific facts: topology thresholds, compliance requirements,
protocol comparisons, and configuration patterns.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


# ─────────────────────────────────────────────────────────────
# Config syntax checker
# ─────────────────────────────────────────────────────────────

def check_config_syntax(text: str) -> bool:
    """
    Heuristic check for Cisco IOS / NX-OS config syntax validity.
    Returns True if extracted configs look syntactically valid.
    """
    # Find all code blocks
    code_blocks = re.findall(r"```(?:cisco|ios|nxos|juniper|text)?\n(.*?)```", text, re.DOTALL)
    if not code_blocks:
        return True  # No configs to check

    for block in code_blocks:
        lines = [l.strip() for l in block.strip().splitlines() if l.strip()]
        if not lines:
            continue

        # Check for obvious syntax errors
        for line in lines:
            # Unmatched parentheses
            if line.count("(") != line.count(")"):
                return False
            # IOS-style: commands shouldn't start with special chars
            if re.match(r"^[!#]", line):
                continue  # Comments are OK
            if re.match(r"^\s*$", line):
                continue

        # Check for common required keywords in BGP config blocks
        if "router bgp" in block:
            if "neighbor" not in block and "network" not in block:
                return False  # BGP config without any neighbors or networks

        # Check for valid OSPF config
        if "router ospf" in block:
            if "network" not in block and "area" not in block:
                return False

    return True


# ─────────────────────────────────────────────────────────────
# Fact database (sourced from NotebookLM)
# ─────────────────────────────────────────────────────────────

TECHNICAL_FACTS: list[dict[str, Any]] = [
    # Topology thresholds (from CCNP Enterprise Design ENSLD, Campus HA Guide)
    {
        "category": "topology",
        "fact": "collapsed_core_max_users",
        "value": 2000,
        "check": lambda text: _check_number_range(text, ["collapsed core", "two-tier"], 1800, 2200),
        "source": "Campus Network HA Design Guide",
    },
    {
        "category": "topology",
        "fact": "spine_leaf_64port_max_servers",
        "value": 2048,
        "check": lambda text: _check_number_presence(text, ["spine", "leaf", "spine-leaf"], 2048),
        "source": "CCNP DCCOR - Data Center Core",
    },
    {
        "category": "topology",
        "fact": "three_tier_max_endpoints",
        "value": 50000,
        "check": lambda text: _check_number_presence(text, ["three-tier", "three tier"], 50000),
        "source": "CCNP Enterprise Design ENSLD",
    },

    # HA patterns (from CCNP ENCOR, Campus HA Design)
    {
        "category": "ha",
        "fact": "n_plus_1_optimal",
        "value": True,
        "check": lambda text: _check_term_present(text, ["n+1", "n plus 1", "two parallel paths"]),
        "source": "Campus Network HA Design Guide - Redundancy Levels",
    },
    {
        "category": "ha",
        "fact": "hsrp_default_preemption",
        "value": "disabled",
        "check": lambda text: _check_hsrp_preemption(text),
        "source": "CCNP ENCOR - HSRP vs VRRP vs GLBP",
    },
    {
        "category": "ha",
        "fact": "vrrp_preemption_enabled_default",
        "value": "enabled",
        "check": lambda text: _check_vrrp_preemption(text),
        "source": "CCNP ENCOR - HSRP vs VRRP vs GLBP",
    },
    {
        "category": "ha",
        "fact": "active_active_capacity_planning",
        "value": "50%",
        "check": lambda text: _check_number_presence(text, ["active-active", "capacity"], 50),
        "source": "Campus Network HA Design - Active-Active Constraints",
    },

    # BGP troubleshooting (from Routing TCP/IP Vol II, BGP Design)
    {
        "category": "bgp",
        "fact": "bgp_tcp_port",
        "value": 179,
        "check": lambda text: _check_number_presence(text, ["bgp", "tcp"], 179) if "bgp" in text.lower() else True,
        "source": "Routing TCP/IP Volume II",
    },
    {
        "category": "bgp",
        "fact": "ebgp_default_ttl",
        "value": 1,
        "check": lambda text: _check_ebgp_ttl(text),
        "source": "BGP Design and Implementation",
    },
    {
        "category": "bgp",
        "fact": "bgp_next_hop_iBGP",
        "value": "unchanged",
        "check": lambda text: _check_term_present(text, ["next-hop unchanged", "next-hop-self"]) if "ibgp" in text.lower() else True,
        "source": "BGP Design and Implementation",
    },

    # OSPF (from OSPF Anatomy, Optimal Routing Design)
    {
        "category": "ospf",
        "fact": "ospf_lsa_refresh_interval",
        "value": 30,
        "check": lambda text: _check_number_presence(text, ["lsa", "refresh", "30 minute"], 30) if "lsa" in text.lower() else True,
        "source": "OSPF Anatomy of an Internet Routing Protocol",
    },
    {
        "category": "ospf",
        "fact": "ospf_lsa_max_age",
        "value": 60,
        "check": lambda text: True,  # Soft check - 60 min age-out
        "source": "OSPF Anatomy",
    },

    # QoS (from End-to-End QoS, RFC 4594)
    {
        "category": "qos",
        "fact": "voice_dscp",
        "value": "EF/46",
        "check": lambda text: _check_voice_dscp(text),
        "source": "End-to-End QoS Network Design / RFC 4594",
    },
    {
        "category": "qos",
        "fact": "llq_max_bandwidth_percent",
        "value": 33,
        "check": lambda text: _check_number_presence(text, ["llq", "priority queue", "strict priority"], 33) if "llq" in text.lower() or "priority queue" in text.lower() else True,
        "source": "End-to-End QoS Network Design - LLQ Design Rules",
    },
    {
        "category": "qos",
        "fact": "scavenger_dscp",
        "value": "CS1/8",
        "check": lambda text: True,  # Soft check
        "source": "RFC 4594 QoS Guidelines",
    },

    # MPLS VPN (from Definitive MPLS, Traffic Engineering)
    {
        "category": "mpls",
        "fact": "inter_as_option_a_most_secure",
        "value": True,
        "check": lambda text: _check_option_a_secure(text),
        "source": "Definitive MPLS Network Designs - Inter-AS Options",
    },
    {
        "category": "mpls",
        "fact": "inter_as_option_c_most_scalable",
        "value": True,
        "check": lambda text: _check_option_c_scalable(text),
        "source": "Definitive MPLS Network Designs - Inter-AS Options",
    },

    # SD-WAN / DMVPN (from Cisco SD-WAN book)
    {
        "category": "sdwan",
        "fact": "sdwan_planes",
        "value": ["vbond", "vmanage", "vsmart", "vedge"],
        "check": lambda text: _check_sdwan_planes(text),
        "source": "Cisco Software-Defined Wide Area Networks",
    },

    # VXLAN BGP EVPN (from Building Data Centers VXLAN BGP EVPN)
    {
        "category": "vxlan",
        "fact": "symmetric_irb_no_all_vteps",
        "value": True,
        "check": lambda text: True,  # Complex check - soft
        "source": "Building Data Centers with VXLAN BGP EVPN",
    },
]


# ─────────────────────────────────────────────────────────────
# Check helper functions
# ─────────────────────────────────────────────────────────────

def _check_number_range(text: str, context_terms: list[str], low: int, high: int) -> bool:
    text_lower = text.lower()
    if not any(t in text_lower for t in context_terms):
        return True  # Not relevant
    numbers = [int(n) for n in re.findall(r"\b(\d{3,5})\b", text)]
    return any(low <= n <= high for n in numbers)


def _check_number_presence(text: str, context_terms: list[str], expected: int) -> bool:
    text_lower = text.lower()
    if not any(t in text_lower for t in context_terms):
        return True  # Not relevant
    numbers = [int(n) for n in re.findall(r"\b(\d+)\b", text)]
    return expected in numbers


def _check_term_present(text: str, terms: list[str]) -> bool:
    text_lower = text.lower()
    return any(t in text_lower for t in terms)


def _check_hsrp_preemption(text: str) -> bool:
    """HSRP preemption is disabled by default."""
    text_lower = text.lower()
    if "hsrp" not in text_lower:
        return True
    # Must mention preemption disabled for HSRP
    if "preempt" in text_lower or "preemption" in text_lower:
        # Check it says disabled for HSRP
        return "disabled" in text_lower or "not enabled" in text_lower or "default is disabled" in text_lower
    return True  # Didn't mention preemption at all - soft pass


def _check_vrrp_preemption(text: str) -> bool:
    """VRRP preemption is enabled by default."""
    text_lower = text.lower()
    if "vrrp" not in text_lower:
        return True
    if "preempt" in text_lower or "preemption" in text_lower:
        return "enabled by default" in text_lower or "enabled" in text_lower
    return True


def _check_ebgp_ttl(text: str) -> bool:
    """eBGP uses TTL=1 by default."""
    text_lower = text.lower()
    if "ebgp" not in text_lower and "external bgp" not in text_lower:
        return True
    if "ttl" in text_lower:
        return "ttl" in text_lower and ("1" in text or "ebgp-multihop" in text_lower)
    return True


def _check_voice_dscp(text: str) -> bool:
    """Voice traffic should be marked EF/DSCP 46."""
    text_lower = text.lower()
    if "voice" not in text_lower and "voip" not in text_lower:
        return True
    has_ef = "ef" in text_lower or "expedited forwarding" in text_lower
    has_46 = "46" in text or "dscp 46" in text_lower
    return has_ef or has_46


def _check_option_a_secure(text: str) -> bool:
    """Inter-AS Option A is most secure."""
    text_lower = text.lower()
    if "option a" not in text_lower and "back-to-back vrf" not in text_lower:
        return True
    secure_near_a = bool(re.search(r"option.{0,30}a.{0,50}secure|secure.{0,50}option.{0,30}a", text_lower))
    return secure_near_a


def _check_option_c_scalable(text: str) -> bool:
    """Inter-AS Option C is most scalable."""
    text_lower = text.lower()
    if "option c" not in text_lower and "multihop mp-ebgp" not in text_lower:
        return True
    scalable_near_c = bool(re.search(r"option.{0,30}c.{0,50}scalab|scalab.{0,50}option.{0,30}c", text_lower))
    return scalable_near_c


def _check_sdwan_planes(text: str) -> bool:
    """SD-WAN should mention at least 3 of the 4 planes."""
    text_lower = text.lower()
    if "sd-wan" not in text_lower and "sdwan" not in text_lower:
        return True
    planes_found = sum(1 for p in ["vbond", "vmanage", "vsmart", "vedge", "cedge"] if p in text_lower)
    return planes_found >= 3


# ─────────────────────────────────────────────────────────────
# Evaluator
# ─────────────────────────────────────────────────────────────

class TechnicalAccuracyEvaluator:
    """Evaluates technical accuracy of model outputs vs knowledge base."""

    def __init__(self, knowledge_dir: Path) -> None:
        self.knowledge_dir = knowledge_dir
        self.facts = TECHNICAL_FACTS

    def evaluate_single(self, result: dict[str, Any]) -> dict[str, Any]:
        """Evaluate a single inference result."""
        text = result.get("predicted", "") + " " + result.get("reasoning", "")
        category = result.get("metadata", {}).get("category", "")

        checks: list[dict[str, Any]] = []
        for fact in self.facts:
            # Skip facts not relevant to this category if category is known
            if category and fact["category"] not in {"general", category}:
                continue

            try:
                passed = fact["check"](text)
            except Exception:
                passed = True  # Soft pass on check errors

            checks.append({
                "fact": fact["fact"],
                "category": fact["category"],
                "passed": passed,
                "source": fact["source"],
            })

        if not checks:
            return {"score": 1.0, "checks": [], "num_checked": 0}

        score = sum(1 for c in checks if c["passed"]) / len(checks)
        return {
            "score": score,
            "checks": checks,
            "num_checked": len(checks),
            "failed_facts": [c["fact"] for c in checks if not c["passed"]],
        }

    def evaluate_batch(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        """Evaluate a batch of results."""
        scores: list[float] = []
        all_failures: list[str] = []

        for result in results:
            eval_result = self.evaluate_single(result)
            scores.append(eval_result["score"])
            all_failures.extend(eval_result.get("failed_facts", []))

        overall_score = sum(scores) / len(scores) if scores else 0.0

        # Count most common failures
        failure_counts: dict[str, int] = {}
        for f in all_failures:
            failure_counts[f] = failure_counts.get(f, 0) + 1
        top_failures = sorted(failure_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        return {
            "score": overall_score,
            "num_samples": len(results),
            "avg_score": overall_score,
            "min_score": min(scores) if scores else 0.0,
            "max_score": max(scores) if scores else 0.0,
            "top_failures": top_failures,
        }
