"""
Live NotebookLM Knowledge Extraction
Uses the NotebookLM MCP server to query the notebook and enrich knowledge JSONs.

This script queries the actual NotebookLM notebook (108 sources, CCDE-level content)
and updates the local knowledge base JSON files with verified, sourced content.

Usage:
    # Full extraction (all query categories)
    python knowledge_extraction/extract_live.py

    # Specific category only
    python knowledge_extraction/extract_live.py --category design

    # Preview queries without saving
    python knowledge_extraction/extract_live.py --dry-run

    # Update a single knowledge file
    python knowledge_extraction/extract_live.py --category compliance --update

IMPORTANT: Requires NotebookLM MCP server to be running.
    Run: nlm login
    Then start: uvx notebooklm-mcp
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent.parent
KNOWLEDGE_DIR = Path(__file__).parent
NOTEBOOK_ID = "86b1a8b9-8a9c-486e-8fb3-2e2e0f914528"

# ─────────────────────────────────────────────────────────────
# Query definitions
# ─────────────────────────────────────────────────────────────

QUERIES: dict[str, list[dict[str, str]]] = {
    "design": [
        {
            "key": "topology_selection_criteria",
            "query": "network topology design patterns - when to use collapsed core vs three-tier vs spine-leaf, with specific user count thresholds and selection criteria",
        },
        {
            "key": "high_availability_patterns",
            "query": "high availability design patterns - redundancy levels, HSRP VRRP GLBP comparison, active-active vs active-standby, N+1 vs N+2",
        },
        {
            "key": "wan_topology_selection",
            "query": "WAN topology selection - hub-and-spoke vs full mesh vs partial mesh, DMVPN vs SD-WAN comparison, WAN edge redundancy",
        },
        {
            "key": "vxlan_evpn_design",
            "query": "VXLAN BGP EVPN data center design - overlay vs underlay, distributed anycast gateway, symmetric vs asymmetric IRB, multi-site interconnect",
        },
        {
            "key": "ccde_methodology",
            "query": "CCDE design methodology - requirements gathering, constraints analysis, trade-off decisions, business requirements to technical design",
        },
    ],
    "routing": [
        {
            "key": "bgp_troubleshooting",
            "query": "BGP troubleshooting methodology - step by step diagnostic for neighbor down, route missing, AS path issues, next-hop unreachable",
        },
        {
            "key": "ospf_design",
            "query": "OSPF design - single area vs multi-area, area types stub NSSA totally stubby, LSA flooding, convergence optimization, BFD, summarization at ABR",
        },
        {
            "key": "mpls_vpn_design",
            "query": "MPLS VPN design - L3VPN vs L2VPN, Route Distinguisher vs Route Target, PE-CE routing options, inter-AS MPLS VPN options A B C",
        },
        {
            "key": "igp_selection",
            "query": "IGP selection criteria - when to use OSPF vs EIGRP vs IS-IS vs BGP, scalability limits, convergence comparison",
        },
    ],
    "qos": [
        {
            "key": "qos_design_end_to_end",
            "query": "QoS design - traffic classification, DSCP marking, queuing strategies CBWFQ LLQ WFQ, policing vs shaping, end-to-end QoS deployment",
        },
        {
            "key": "dscp_markings",
            "query": "DSCP marking guidelines - voice EF, video AF, call signaling CS3, scavenger CS1, best-effort per RFC 4594",
        },
    ],
    "security": [
        {
            "key": "security_architecture",
            "query": "network security architecture - zone-based firewall design, DMZ zones, firewall placement in campus network, security segmentation with VLANs and VRFs",
        },
        {
            "key": "private_vlans",
            "query": "Private VLAN design - isolated ports, community ports, promiscuous ports, use cases for healthcare and data center segmentation",
        },
    ],
    "cloud": [
        {
            "key": "cloud_connectivity",
            "query": "cloud connectivity design - AWS Direct Connect vs VPN, hybrid cloud architecture, SD-WAN to cloud integration",
        },
    ],
}

# ─────────────────────────────────────────────────────────────
# MCP client
# ─────────────────────────────────────────────────────────────

def query_notebooklm_mcp(query: str, notebook_id: str = NOTEBOOK_ID) -> str | None:
    """
    Query NotebookLM via the MCP CLI tool.
    Returns the answer text or None on failure.
    """
    try:
        # Use the nlm CLI to query the notebook
        result = subprocess.run(
            ["nlm", "query", "--notebook", notebook_id, "--query", query],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode == 0 and result.stdout:
            return result.stdout.strip()
        else:
            print(f"  MCP query failed: {result.stderr[:200]}")
            return None
    except subprocess.TimeoutExpired:
        print("  Query timed out after 120s")
        return None
    except FileNotFoundError:
        print("  nlm CLI not found. Install: pip install notebooklm-mcp-cli")
        return None
    except Exception as e:
        print(f"  Query error: {e}")
        return None


# ─────────────────────────────────────────────────────────────
# Knowledge enrichment
# ─────────────────────────────────────────────────────────────

CATEGORY_TO_FILE: dict[str, str] = {
    "design": "design_patterns.json",
    "routing": "troubleshooting_trees.json",
    "qos": "vendor_specifics.json",  # QoS goes in vendor specifics
    "security": "design_patterns.json",  # Security patterns in design
    "cloud": "cost_benchmarks.json",
}

CATEGORY_TO_SECTION: dict[str, str] = {
    "design": "topology_patterns",
    "routing": "bgp_issues",
    "qos": "cisco",
    "security": "security_patterns",
    "cloud": "cloud_benchmarks",
}


def enrich_knowledge_file(
    file_path: Path,
    section: str,
    key: str,
    answer: str,
    query: str,
) -> bool:
    """Add extracted knowledge to the appropriate JSON file."""
    if not file_path.exists():
        print(f"  File not found: {file_path}")
        return False

    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)

    # Navigate to section
    if section not in data:
        data[section] = {}

    # Add the enriched entry
    entry: dict[str, Any] = {
        "_source": "NotebookLM extraction",
        "_notebook_id": NOTEBOOK_ID,
        "_query": query,
        "_extracted_at": datetime.now().isoformat(),
        "description": answer[:2000],  # First 2000 chars
        "full_content": answer,
    }

    data[section][key] = entry

    # Update metadata
    if "metadata" in data:
        data["metadata"]["last_updated"] = datetime.now().strftime("%Y-%m-%d")
        data["metadata"]["live_extractions"] = data["metadata"].get("live_extractions", 0) + 1

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return True


# ─────────────────────────────────────────────────────────────
# Main extraction
# ─────────────────────────────────────────────────────────────

def extract_category(
    category: str,
    dry_run: bool = False,
    pause_seconds: float = 1.0,
) -> dict[str, Any]:
    """Extract all queries for a category."""
    queries = QUERIES.get(category, [])
    if not queries:
        print(f"Unknown category: {category}. Available: {list(QUERIES.keys())}")
        return {"category": category, "success": 0, "failed": 0}

    file_name = CATEGORY_TO_FILE.get(category, "design_patterns.json")
    section = CATEGORY_TO_SECTION.get(category, "notebooklm_extractions")
    file_path = KNOWLEDGE_DIR / file_name

    print(f"\n[{category.upper()}] → {file_name}")
    success = 0
    failed = 0

    for q in queries:
        key = q["key"]
        query = q["query"]
        print(f"  Query: {query[:80]}...")

        if dry_run:
            print(f"  [DRY RUN] Would save to {file_name} § {section}.{key}")
            success += 1
            continue

        answer = query_notebooklm_mcp(query)
        if answer:
            if enrich_knowledge_file(file_path, section, key, answer, query):
                print(f"  ✓ Saved to {section}.{key} ({len(answer)} chars)")
                success += 1
            else:
                failed += 1
        else:
            print(f"  ✗ No answer received")
            failed += 1

        time.sleep(pause_seconds)

    return {"category": category, "success": success, "failed": failed}


def run_full_extraction(dry_run: bool = False) -> None:
    """Run extraction for all categories."""
    print("=" * 60)
    print(f"NotebookLM Live Knowledge Extraction")
    print(f"Notebook: {NOTEBOOK_ID}")
    print(f"Mode: {'DRY RUN' if dry_run else 'LIVE'}")
    print("=" * 60)

    total_success = 0
    total_failed = 0

    for category in QUERIES:
        result = extract_category(category, dry_run=dry_run)
        total_success += result["success"]
        total_failed += result["failed"]

    print(f"\n{'='*60}")
    print(f"Extraction complete!")
    print(f"  Successful: {total_success}")
    print(f"  Failed: {total_failed}")
    print(f"{'='*60}")

    if not dry_run:
        # Run coverage report
        print("\nKnowledge coverage after extraction:")
        sys.path.insert(0, str(ROOT))
        from knowledge_extraction.extract_from_notebooklm import NotebookLMKnowledgeExtractor
        extractor = NotebookLMKnowledgeExtractor()
        report = extractor.generate_knowledge_coverage_report()
        print(f"  Coverage score: {report['coverage_score']}/100")
        print(f"  Total index terms: {report['total_index_terms']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live NotebookLM Knowledge Extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--category",
        choices=list(QUERIES.keys()) + ["all"],
        default="all",
        help="Category to extract (default: all)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview without saving")
    parser.add_argument("--pause", type=float, default=1.0, help="Seconds between queries")
    args = parser.parse_args()

    if args.category == "all":
        run_full_extraction(dry_run=args.dry_run)
    else:
        result = extract_category(args.category, dry_run=args.dry_run, pause_seconds=args.pause)
        print(f"\nResult: {result['success']} saved, {result['failed']} failed")


if __name__ == "__main__":
    main()
