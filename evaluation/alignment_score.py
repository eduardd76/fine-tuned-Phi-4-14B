"""
NotebookLM Alignment Score

Measures how well model outputs align with the NotebookLM knowledge base.
Checks:
1. Source attribution - are book titles/chapters mentioned?
2. Fact consistency - do numbers and facts match the knowledge base?
3. Compliance specificity - are exact requirement sections cited?
4. Protocol accuracy - are protocol behaviors described correctly?
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# Source books in the NotebookLM notebook
NOTEBOOKLM_SOURCES = {
    "enterprise_design": [
        "enterprise network design",
        "ccnp enterprise design",
        "ensld",
        "designing cisco network service architectures",
        "campus network",
    ],
    "routing": [
        "routing tcp/ip",
        "tcp/ip volume",
        "jeff doyle",
        "bgp design and implementation",
        "optimal routing design",
        "ospf anatomy",
    ],
    "data_center": [
        "vxlan bgp evpn",
        "building data centers",
        "ccnp dccor",
        "data center core",
        "trill fabricpath",
    ],
    "security": [
        "cisco asa",
        "network security architectures",
        "end to end network security",
        "firewall and ips design",
        "cvd firewall",
    ],
    "qos": [
        "end-to-end qos",
        "qos network design",
        "rfc 4594",
        "qos end to end",
    ],
    "mpls": [
        "definitive mpls",
        "mpls design",
        "traffic engineering with mpls",
        "advanced mpls",
        "inter as mpls",
    ],
    "sd_wan": [
        "cisco software-defined wide area",
        "sd-wan",
        "dmvpn",
    ],
    "ccde": [
        "ccde",
        "orhan ergun",
        "art of network architecture",
        "patterns in network",
    ],
    "cloud": [
        "aws architect",
        "cloud computing",
        "hybrid cloud",
        "azure",
    ],
}

# Knowledge facts extracted from NotebookLM for alignment checking
ALIGNMENT_FACTS: list[dict[str, Any]] = [
    # Sourced from notebook query: topology design patterns
    {
        "domain": "topology",
        "claim": "collapsed_core_50_to_100_branch",
        "patterns": [r"50.{0,10}100.{0,30}branch|branch.{0,30}50.{0,10}100"],
        "required_context": ["collapsed core", "two-tier"],
        "source": "Campus Network HA Design Guide",
    },
    {
        "domain": "topology",
        "claim": "spine_leaf_east_west",
        "patterns": [r"east.{0,10}west|east-west"],
        "required_context": ["spine", "leaf"],
        "source": "CCNP DCCOR - Spine-Leaf Traffic Patterns",
    },

    # Sourced from: BGP troubleshooting methodology
    {
        "domain": "bgp",
        "claim": "bgp_next_hop_requirement",
        "patterns": [r"next.hop.{0,30}unreachable|next.hop.{0,30}inaccessible"],
        "required_context": ["bgp", "route"],
        "source": "BGP Design and Implementation - Next-Hop Resolution",
    },
    {
        "domain": "bgp",
        "claim": "bgp_network_exact_match",
        "patterns": [r"exact match|exact prefix|network.{0,30}routing table"],
        "required_context": ["bgp", "network command"],
        "source": "BGP Design and Implementation - Route Origination",
    },

    # Sourced from: OSPF design
    {
        "domain": "ospf",
        "claim": "totally_stubby_blocks_type3",
        "patterns": [r"type.?3.{0,30}block|block.{0,30}type.?3"],
        "required_context": ["totally stubby", "nssa"],
        "source": "OSPF Anatomy / Optimal Routing Design",
    },
    {
        "domain": "ospf",
        "claim": "abr_null0_discard",
        "patterns": [r"null.?0.{0,30}discard|discard.{0,30}null"],
        "required_context": ["abr", "summarization"],
        "source": "OSPF Anatomy - Route Summarization",
    },

    # Sourced from: QoS design
    {
        "domain": "qos",
        "claim": "classify_at_edge",
        "patterns": [r"classify.{0,30}edge|edge.{0,30}classify|trust boundary"],
        "required_context": ["qos", "classification"],
        "source": "End-to-End QoS Network Design - Classification",
    },
    {
        "domain": "qos",
        "claim": "shaping_buffers_policing_drops",
        "patterns": [r"shaping.{0,50}buffer|policing.{0,50}drop|buffer.{0,50}shaping"],
        "required_context": ["qos"],
        "source": "End-to-End QoS Network Design - Policing vs Shaping",
    },

    # Sourced from: MPLS VPN
    {
        "domain": "mpls",
        "claim": "rd_uniqueness_rt_distribution",
        "patterns": [r"rd.{0,30}unique|route distinguisher.{0,30}unique"],
        "required_context": ["mpls", "vpn"],
        "source": "Definitive MPLS - RD vs RT",
    },
    {
        "domain": "mpls",
        "claim": "vrf_isolation",
        "patterns": [r"vrf.{0,30}isolat|separate routing table|independent routing"],
        "required_context": ["vrf"],
        "source": "Definitive MPLS / Network Security Architectures",
    },

    # Sourced from: HA patterns
    {
        "domain": "ha",
        "claim": "fhrp_default_gateway",
        "patterns": [r"first.hop.redundancy|fhrp|default gateway.{0,30}redundanc"],
        "required_context": ["ha", "redundancy", "gateway"],
        "source": "Campus Network HA Design - FHRP Overview",
    },
    {
        "domain": "ha",
        "claim": "glbp_multiple_vms",
        "patterns": [r"multiple.{0,30}virtual mac|avg|avf|active virtual"],
        "required_context": ["glbp"],
        "source": "CCNP ENCOR - GLBP Operation",
    },

    # Sourced from: SD-WAN
    {
        "domain": "sdwan",
        "claim": "bfd_application_aware",
        "patterns": [r"bfd.{0,50}latency|bfd.{0,50}jitter|application.aware.routing"],
        "required_context": ["sd-wan", "bfd"],
        "source": "Cisco SD-WAN - Application-Aware Routing",
    },
    {
        "domain": "sdwan",
        "claim": "full_mesh_formula",
        "patterns": [r"n.{0,5}\(.{0,5}n.{0,5}-.{0,5}1.{0,5}\).{0,10}/\s*2|n\(n-1\)/2"],
        "required_context": ["full mesh", "topology"],
        "source": "CCNP Enterprise Design - WAN Topology Selection",
    },

    # Sourced from: VXLAN
    {
        "domain": "vxlan",
        "claim": "vtep_underlay_ip_reachability",
        "patterns": [r"vtep.{0,30}ip|underlay.{0,30}vtep|ospf.{0,30}underlay"],
        "required_context": ["vxlan", "vtep"],
        "source": "Building Data Centers with VXLAN BGP EVPN",
    },
    {
        "domain": "vxlan",
        "claim": "vxlan_50_byte_overhead",
        "patterns": [r"50.{0,20}byte|50 byte overhead|vxlan.{0,30}overhead"],
        "required_context": ["vxlan", "mtu"],
        "source": "Building Data Centers with VXLAN BGP EVPN - MTU",
    },

    # Sourced from: CCDE design methodology
    {
        "domain": "methodology",
        "claim": "business_drives_technology",
        "patterns": [r"business.{0,30}drive|business.{0,30}technolog|top.?down"],
        "required_context": ["design", "requirement"],
        "source": "Art of Network Architecture / CCDE In Depth",
    },
]


class AlignmentScoreEvaluator:
    """Measures how well outputs align with NotebookLM knowledge base."""

    def __init__(self, knowledge_dir: Path) -> None:
        self.knowledge_dir = knowledge_dir
        self.source_keywords = {
            book: keywords for book, keywords in NOTEBOOKLM_SOURCES.items()
        }
        self.facts = ALIGNMENT_FACTS

    def check_source_attribution(self, text: str) -> dict[str, Any]:
        """
        Check if text references NotebookLM sources (book titles, authors, etc.).
        Returns attribution coverage score.
        """
        text_lower = text.lower()
        domains_cited: set[str] = set()

        for domain, keywords in self.source_keywords.items():
            if any(kw in text_lower for kw in keywords):
                domains_cited.add(domain)

        # Count direct source citations
        citation_count = len(re.findall(
            r"\[Source:|Source:|from .{3,60} chapter|per .{3,60} §|according to .{3,60}",
            text,
            re.IGNORECASE,
        ))

        # Check for compliance version specificity
        compliance_versions = re.findall(
            r"PCI.DSS\s+\d+\.\d+|HIPAA\s+\d{4}|NIST\s+CSF\s+\d+\.\d+|ISO\s+27001:\d{4}",
            text,
            re.IGNORECASE,
        )

        return {
            "domains_cited": list(domains_cited),
            "domain_coverage": len(domains_cited),
            "citation_count": citation_count,
            "compliance_versions": compliance_versions,
            "has_specific_versions": len(compliance_versions) > 0,
        }

    def check_fact_alignment(self, text: str) -> dict[str, Any]:
        """Check if specific facts from NotebookLM are correctly represented."""
        text_lower = text.lower()
        applicable_facts = 0
        aligned_facts = 0
        failed_facts: list[str] = []

        for fact in self.facts:
            # Check if fact is applicable (context terms present)
            if not any(ctx in text_lower for ctx in fact["required_context"]):
                continue

            applicable_facts += 1

            # Check if fact pattern is present
            matched = any(
                re.search(pattern, text_lower)
                for pattern in fact["patterns"]
            )

            if matched:
                aligned_facts += 1
            else:
                failed_facts.append(f"{fact['domain']}/{fact['claim']}")

        if applicable_facts == 0:
            return {"score": 1.0, "applicable": 0, "aligned": 0, "failed": []}

        score = aligned_facts / applicable_facts
        return {
            "score": score,
            "applicable": applicable_facts,
            "aligned": aligned_facts,
            "failed": failed_facts,
        }

    def evaluate_single(self, result: dict[str, Any]) -> dict[str, Any]:
        """Evaluate alignment for a single result."""
        text = result.get("reasoning", "") + " " + result.get("predicted", "")

        attribution = self.check_source_attribution(text)
        fact_align = self.check_fact_alignment(text)

        # Composite score:
        # 50% fact alignment + 30% domain coverage + 20% citation quality
        max_domains = len(self.source_keywords)
        domain_score = min(1.0, attribution["domain_coverage"] / 3)  # At least 3 domains = full score
        citation_score = min(1.0, attribution["citation_count"] / 5)  # 5+ citations = full score

        composite = (
            fact_align["score"] * 0.50
            + domain_score * 0.30
            + citation_score * 0.20
        )

        return {
            "score": composite,
            "attribution": attribution,
            "fact_alignment": fact_align,
        }

    def evaluate_batch(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        """Evaluate alignment across a batch."""
        scores: list[float] = []
        all_failures: list[str] = []

        for result in results:
            eval_result = self.evaluate_single(result)
            scores.append(eval_result["score"])
            all_failures.extend(eval_result["fact_alignment"].get("failed", []))

        avg_score = sum(scores) / len(scores) if scores else 0.0

        failure_counts: dict[str, int] = {}
        for f in all_failures:
            failure_counts[f] = failure_counts.get(f, 0) + 1
        top_failures = sorted(failure_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        return {
            "score": avg_score,
            "num_samples": len(results),
            "min_score": min(scores) if scores else 0.0,
            "max_score": max(scores) if scores else 0.0,
            "top_alignment_failures": top_failures,
        }
