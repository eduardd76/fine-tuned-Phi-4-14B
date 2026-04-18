"""
NotebookLM Knowledge Extraction Helper

Since NotebookLM is a web UI (https://notebooklm.google.com) that cannot be
programmatically queried, this module:
1. Loads pre-extracted knowledge from the local JSON files
2. Provides a query interface to search the local knowledge base
3. Includes a CLI to manually add new knowledge from NotebookLM queries

Usage:
    # Query the knowledge base
    python extract_from_notebooklm.py query --topic "BGP troubleshooting"

    # Add new knowledge from a manual NotebookLM query
    python extract_from_notebooklm.py add --category design_patterns --key spine_leaf_variant

    # Generate coverage report
    python extract_from_notebooklm.py report

    # List all available categories and keys
    python extract_from_notebooklm.py list
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table
from rich.tree import Tree

console = Console()

_CONTAMINATION_PATTERNS = [
    r'<\|repo_name\|>', r'<\|file_sep\|>', r'\[\d+\]:\s*#!/',
    r'#!/usr/bin/env\s+python', r'Copyright \d{4}.*?(?:CSIRO|Data61)',
    r'</system\s*\nuser', r'<\|im_start\|>user',
    r'github\.com/[a-zA-Z0-9\-]+/[a-zA-Z0-9\-]+', r'\bpip install\b',
]
_NETWORKING_TERMS = re.compile(
    r'\b(?:BGP|OSPF|MPLS|VXLAN|QoS|DSCP|VLAN|VRF|ACL|firewall|router|switch|subnet|'
    r'routing|protocol|interface|topology|bandwidth|latency|MTU|TCP|UDP|IP|WAN|LAN)\b',
    re.IGNORECASE,
)


def clean_extracted_text(text: str) -> str:
    """Strip contamination artifacts from extracted text."""
    for pattern in _CONTAMINATION_PATTERNS:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)
    return text.strip()


def is_networking_content(text: str) -> bool:
    """Return True if text contains at least 2 distinct networking terms."""
    return len(set(_NETWORKING_TERMS.findall(text.upper()))) >= 2


KNOWLEDGE_DIR = Path(__file__).parent
KNOWLEDGE_FILES = {
    "design_patterns": KNOWLEDGE_DIR / "design_patterns.json",
    "troubleshooting_trees": KNOWLEDGE_DIR / "troubleshooting_trees.json",
    "compliance_requirements": KNOWLEDGE_DIR / "compliance_requirements.json",
    "vendor_specifics": KNOWLEDGE_DIR / "vendor_specifics.json",
    "configuration_templates": KNOWLEDGE_DIR / "configuration_templates.json",
    "cost_benchmarks": KNOWLEDGE_DIR / "cost_benchmarks.json",
}


class NotebookLMKnowledgeExtractor:
    """
    Extracts and structures knowledge for dataset generation.

    Since NotebookLM is a web UI, this class loads pre-populated knowledge
    from JSON files that represent CCDE-level industry standards. Users can
    manually add knowledge from NotebookLM queries via the CLI.

    Attributes:
        knowledge_base: Loaded knowledge from all JSON files
        query_index: Flat index for fast keyword search
    """

    def __init__(self) -> None:
        self.knowledge_base: dict[str, Any] = {}
        self.query_index: dict[str, list[str]] = {}
        self._load_all()
        self._build_index()

    def _load_all(self) -> None:
        """Load all knowledge JSON files into memory."""
        for category, path in KNOWLEDGE_FILES.items():
            if path.exists():
                with open(path, encoding="utf-8") as f:
                    self.knowledge_base[category] = json.load(f)
                console.print(
                    f"[green]Loaded[/green] {category} "
                    f"({path.stat().st_size // 1024}KB)"
                )
            else:
                console.print(f"[yellow]Missing[/yellow] {path.name}")
                self.knowledge_base[category] = {}

    def _build_index(self) -> None:
        """Build a flat keyword index for fast lookup across all knowledge."""
        self.query_index = {}

        def _index_value(value: Any, path: str) -> None:
            if isinstance(value, str):
                for word in value.lower().split():
                    cleaned = word.strip(".,;:()[]{}\"'")
                    if len(cleaned) > 3:
                        self.query_index.setdefault(cleaned, []).append(path)
            elif isinstance(value, dict):
                for k, v in value.items():
                    _index_value(v, f"{path}.{k}")
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    _index_value(item, f"{path}[{i}]")

        for category, data in self.knowledge_base.items():
            _index_value(data, category)

    def extract_design_patterns(self) -> dict[str, Any]:
        """
        Return structured design pattern knowledge.

        Returns:
            Dictionary with topology_patterns, wan_patterns, security_patterns, ha_patterns
        """
        patterns = self.knowledge_base.get("design_patterns", {})
        return {
            "topology_patterns": patterns.get("topology_patterns", {}),
            "wan_patterns": patterns.get("wan_patterns", {}),
            "security_patterns": patterns.get("security_patterns", {}),
            "ha_patterns": patterns.get("ha_patterns", {}),
            "data_center_patterns": patterns.get("data_center_patterns", {}),
        }

    def extract_troubleshooting_methodologies(self) -> dict[str, Any]:
        """
        Return structured troubleshooting decision trees.

        Returns:
            Dictionary with BGP, OSPF, interface, QoS, and connectivity issue trees
        """
        trees = self.knowledge_base.get("troubleshooting_trees", {})
        return {
            "bgp_issues": trees.get("bgp_issues", {}),
            "ospf_issues": trees.get("ospf_issues", {}),
            "interface_issues": trees.get("interface_issues", {}),
            "qos_issues": trees.get("qos_issues", {}),
            "connectivity_sequence": trees.get("connectivity_issues", {}),
            "routing_loops": trees.get("routing_loops", {}),
        }

    def extract_compliance_requirements(self) -> dict[str, Any]:
        """
        Return structured compliance requirements.

        Returns:
            Dictionary with PCI-DSS, HIPAA, SOX, NIST CSF requirements
        """
        compliance = self.knowledge_base.get("compliance_requirements", {})
        return {
            "pci_dss_v4": compliance.get("pci_dss", {}),
            "hipaa": compliance.get("hipaa", {}),
            "sox": compliance.get("sox", {}),
            "nist_csf_v2": compliance.get("nist_csf", {}),
            "iso_27001_2022": compliance.get("iso_27001", {}),
        }

    def extract_vendor_specifics(self) -> dict[str, Any]:
        """
        Return vendor-specific configuration patterns and platform details.

        Returns:
            Dictionary with Cisco, Juniper, Arista, and other vendor data
        """
        vendors = self.knowledge_base.get("vendor_specifics", {})
        return {
            "cisco": vendors.get("cisco", {}),
            "juniper": vendors.get("juniper", {}),
            "arista": vendors.get("arista", {}),
            "palo_alto": vendors.get("palo_alto", {}),
            "f5": vendors.get("f5", {}),
        }

    def extract_configuration_templates(self) -> dict[str, Any]:
        """Return actual IOS/NX-OS/JunOS configuration templates."""
        return self.knowledge_base.get("configuration_templates", {})

    def extract_cost_benchmarks(self) -> dict[str, Any]:
        """Return CapEx/OpEx benchmarks by network scale."""
        benchmarks = self.knowledge_base.get("cost_benchmarks", {})
        return {
            "capex_by_scale": benchmarks.get("capex_by_scale", {}),
            "opex_models": benchmarks.get("opex_models", {}),
            "tco_models": benchmarks.get("tco_models", {}),
            "roi_considerations": benchmarks.get("roi_considerations", {}),
        }

    def query(self, topic: str) -> dict[str, Any]:
        """
        Search the knowledge base for a given topic.

        Args:
            topic: Search term or phrase (e.g., 'BGP troubleshooting', 'PCI-DSS')

        Returns:
            Dictionary of matching knowledge entries organized by category
        """
        results: dict[str, Any] = {}
        search_terms = [t.lower().strip(".,;:()") for t in topic.split() if len(t) > 2]

        for term in search_terms:
            matching_paths = self.query_index.get(term, [])
            for path in matching_paths:
                category = path.split(".")[0]
                results.setdefault(category, {"matches": [], "paths": []})
                results[category]["paths"].append(path)

        # Resolve top-level sections for matched categories
        for category in list(results.keys()):
            data = self.knowledge_base.get(category, {})
            # Find top-level sections that contain matches
            relevant_sections = {}
            for path in results[category]["paths"]:
                parts = path.split(".")
                if len(parts) >= 2:
                    section_key = parts[1]
                    if section_key in data:
                        relevant_sections[section_key] = data[section_key]
            results[category]["data"] = relevant_sections
            results[category]["match_count"] = len(set(results[category]["paths"]))

        return results

    def add_knowledge(
        self,
        category: str,
        key: str,
        value: Any,
        source: str = "Manual NotebookLM extraction",
    ) -> bool:
        """
        Add new knowledge from a manual NotebookLM query.

        Args:
            category: Knowledge category (e.g., 'design_patterns')
            key: Key within the category (e.g., 'new_topology')
            value: The knowledge data to store
            source: Description of the source (for attribution)

        Returns:
            True if successfully added, False otherwise
        """
        if category not in KNOWLEDGE_FILES:
            console.print(f"[red]Unknown category: {category}[/red]")
            console.print(f"Valid categories: {list(KNOWLEDGE_FILES.keys())}")
            return False

        path = KNOWLEDGE_FILES[category]
        if not path.exists():
            console.print(f"[red]File not found: {path}[/red]")
            return False

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        # Clean value of contamination artifacts before storing
        if isinstance(value, str):
            value = clean_extracted_text(value)

        # Add metadata to the new entry
        entry = {
            "_metadata": {
                "source": source,
                "added": datetime.now().isoformat(),
                "extraction_method": "manual_notebooklm"
            }
        }
        if isinstance(value, dict):
            entry.update(value)
        else:
            entry["value"] = value

        # Navigate to appropriate section and add
        # For design_patterns, add to topology_patterns by default
        top_level_map = {
            "design_patterns": "topology_patterns",
            "troubleshooting_trees": "bgp_issues",
            "compliance_requirements": "pci_dss",
            "vendor_specifics": "cisco",
            "configuration_templates": "bgp_templates",
            "cost_benchmarks": "capex_by_scale",
        }
        section = top_level_map.get(category, list(data.keys())[1])

        if section in data and isinstance(data[section], dict):
            data[section][key] = entry
        else:
            data[key] = entry

        # Update metadata
        if "metadata" in data:
            data["metadata"]["last_updated"] = datetime.now().strftime("%Y-%m-%d")

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        # Reload knowledge base
        self._load_all()
        self._build_index()

        console.print(f"[green]Added[/green] {key} to {category}")
        return True

    def generate_knowledge_coverage_report(self) -> dict[str, Any]:
        """
        Generate a report on knowledge base coverage and statistics.

        Returns:
            Dictionary with coverage statistics per category
        """
        report: dict[str, Any] = {
            "generated_at": datetime.now().isoformat(),
            "total_categories": len(self.knowledge_base),
            "total_index_terms": len(self.query_index),
            "categories": {}
        }

        for category, data in self.knowledge_base.items():
            top_level_keys = [k for k in data.keys() if not k.startswith("_")]
            entry_count = self._count_entries(data)
            report["categories"][category] = {
                "file": str(KNOWLEDGE_FILES.get(category, "unknown")),
                "top_level_sections": top_level_keys,
                "section_count": len(top_level_keys),
                "total_entries": entry_count,
                "has_metadata": "metadata" in data,
            }

        report["coverage_score"] = self._calculate_coverage_score()
        return report

    def _count_entries(self, data: Any, depth: int = 0) -> int:
        """Recursively count leaf entries in a nested structure."""
        if depth > 5:
            return 1
        if isinstance(data, dict):
            return sum(self._count_entries(v, depth + 1) for v in data.values())
        if isinstance(data, list):
            return len(data)
        return 1

    def _calculate_coverage_score(self) -> float:
        """Calculate overall knowledge coverage score (0-100)."""
        required_categories = list(KNOWLEDGE_FILES.keys())
        present = sum(
            1 for c in required_categories
            if self.knowledge_base.get(c, {})
        )
        base_score = (present / len(required_categories)) * 60

        # Bonus for depth of coverage
        total_terms = len(self.query_index)
        depth_score = min(40.0, total_terms / 100)

        return round(base_score + depth_score, 1)


def cmd_query(args: argparse.Namespace, extractor: NotebookLMKnowledgeExtractor) -> None:
    """Handle query command."""
    results = extractor.query(args.topic)

    if not results:
        console.print(f"[yellow]No results found for: {args.topic}[/yellow]")
        return

    console.print(f"\n[bold]Query results for: '{args.topic}'[/bold]\n")
    for category, info in results.items():
        console.print(f"[cyan]{category}[/cyan] ({info['match_count']} matches)")
        for section_key in list(info.get("data", {}).keys())[:3]:
            console.print(f"  - {section_key}")
    console.print()


def cmd_add(args: argparse.Namespace, extractor: NotebookLMKnowledgeExtractor) -> None:
    """Handle add command - prompt for value from stdin."""
    console.print(f"\n[bold]Adding new knowledge to {args.category}/{args.key}[/bold]")
    console.print("Paste JSON value (end with EOF / Ctrl+D on Unix, Ctrl+Z on Windows):")

    lines = []
    try:
        for line in sys.stdin:
            lines.append(line)
    except EOFError:
        pass

    raw_input = "".join(lines).strip()
    if not raw_input:
        console.print("[red]No value provided[/red]")
        return

    try:
        value = json.loads(raw_input)
    except json.JSONDecodeError:
        # Treat as plain string
        value = raw_input

    source = args.source or "Manual NotebookLM extraction"
    extractor.add_knowledge(args.category, args.key, value, source)


def cmd_report(args: argparse.Namespace, extractor: NotebookLMKnowledgeExtractor) -> None:
    """Handle report command."""
    report = extractor.generate_knowledge_coverage_report()

    console.print("\n[bold]Knowledge Coverage Report[/bold]")
    console.print(f"Generated: {report['generated_at']}")
    console.print(f"Coverage Score: [green]{report['coverage_score']}/100[/green]")
    console.print(f"Total Index Terms: {report['total_index_terms']}\n")

    table = Table(title="Category Coverage")
    table.add_column("Category", style="cyan")
    table.add_column("Sections", justify="right")
    table.add_column("Entries", justify="right")
    table.add_column("Has Metadata")

    for cat, info in report["categories"].items():
        table.add_row(
            cat,
            str(info["section_count"]),
            str(info["total_entries"]),
            "[green]Yes[/green]" if info["has_metadata"] else "[red]No[/red]",
        )

    console.print(table)

    # Write report to file
    report_path = KNOWLEDGE_DIR / "knowledge_coverage_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    console.print(f"\nReport saved to {report_path}")


def cmd_list(args: argparse.Namespace, extractor: NotebookLMKnowledgeExtractor) -> None:
    """Handle list command."""
    tree = Tree("[bold]Knowledge Base Structure[/bold]")

    for category, data in extractor.knowledge_base.items():
        branch = tree.add(f"[cyan]{category}[/cyan]")
        if isinstance(data, dict):
            for key in list(data.keys())[:10]:
                if not key.startswith("_"):
                    branch.add(key)
            remaining = len(data) - 10
            if remaining > 0:
                branch.add(f"... and {remaining} more")

    console.print(tree)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NotebookLM Knowledge Extraction Helper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # query command
    query_parser = subparsers.add_parser("query", help="Search the knowledge base")
    query_parser.add_argument("--topic", required=True, help="Topic to search for")

    # add command
    add_parser = subparsers.add_parser("add", help="Add new knowledge from NotebookLM")
    add_parser.add_argument(
        "--category",
        required=True,
        choices=list(KNOWLEDGE_FILES.keys()),
        help="Knowledge category",
    )
    add_parser.add_argument("--key", required=True, help="Key for the new entry")
    add_parser.add_argument(
        "--source",
        default=None,
        help="Source description (default: 'Manual NotebookLM extraction')",
    )

    # report command
    subparsers.add_parser("report", help="Generate knowledge coverage report")

    # list command
    subparsers.add_parser("list", help="List all categories and keys")

    args = parser.parse_args()

    console.print("[bold blue]NotebookLM Knowledge Extractor[/bold blue]")
    console.print(f"Loading knowledge from: {KNOWLEDGE_DIR}\n")

    extractor = NotebookLMKnowledgeExtractor()

    if args.command == "query":
        cmd_query(args, extractor)
    elif args.command == "add":
        cmd_add(args, extractor)
    elif args.command == "report":
        cmd_report(args, extractor)
    elif args.command == "list":
        cmd_list(args, extractor)


if __name__ == "__main__":
    main()
