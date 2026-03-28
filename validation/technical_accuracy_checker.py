"""
Technical Accuracy Checker

Rule-based validation of network design and troubleshooting responses.
Checks protocol usage, IP addressing, topology appropriateness,
and compliance completeness.
"""

from __future__ import annotations

import ipaddress
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console

console = Console()

sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class AccuracyCheckResult:
    """Result of a single accuracy check."""
    check_name: str
    passed: bool
    severity: str  # "error", "warning", "info"
    message: str
    evidence: str = ""


@dataclass
class SampleAccuracyReport:
    """Aggregate accuracy report for a sample."""
    sample_index: int
    checks: list[AccuracyCheckResult] = field(default_factory=list)
    overall_score: float = 0.0
    critical_failures: int = 0

    def add_check(self, result: AccuracyCheckResult) -> None:
        self.checks.append(result)

    def calculate_score(self) -> None:
        if not self.checks:
            self.overall_score = 100.0
            return
        error_checks = [c for c in self.checks if c.severity == "error"]
        self.critical_failures = sum(1 for c in error_checks if not c.passed)
        total = len(self.checks)
        passed = sum(1 for c in self.checks if c.passed)
        self.overall_score = (passed / total) * 100


# Protocol usage rules
PROTOCOL_RULES = {
    "bgp": {
        "appropriate_for": ["wan", "multi-site", "internet edge", "data center"],
        "inappropriate_for": ["small campus < 500 users as sole routing"],
        "keywords": ["bgp", "as-path", "as number", "autonomous system", "peer"],
    },
    "ospf": {
        "appropriate_for": ["campus", "enterprise campus", "single area", "multi-area"],
        "inappropriate_for": ["inter-AS routing", "internet routing"],
        "keywords": ["ospf", "link state", "lsa", "area", "spf"],
    },
    "eigrp": {
        "appropriate_for": ["cisco-only environments", "campus", "branch"],
        "inappropriate_for": ["multi-vendor environments", "service providers"],
        "keywords": ["eigrp", "diffusing update algorithm", "dual algorithm"],
    },
    "vxlan": {
        "appropriate_for": ["data center", "spine-leaf", "overlay networks"],
        "inappropriate_for": ["campus access layer", "small branch"],
        "keywords": ["vxlan", "vtep", "vni", "evpn"],
    },
}

# Topology appropriateness rules
TOPOLOGY_RULES = {
    "spine_leaf": {
        "requires_context": ["data center", "server", "east-west", "dc"],
        "not_for": ["campus user", "branch", "wan"],
        "error_msg": "Spine-leaf is a DC topology, not campus/branch",
    },
    "three_tier": {
        "appropriate_user_count_min": 2000,
        "note": "Three-tier for large enterprises",
    },
    "collapsed_core": {
        "max_user_count": 2000,
        "note": "Collapsed core for small/medium networks only",
    },
}

# Valid RFC1918 ranges
RFC1918_RANGES = [
    ipaddress.IPv4Network("10.0.0.0/8"),
    ipaddress.IPv4Network("172.16.0.0/12"),
    ipaddress.IPv4Network("192.168.0.0/16"),
]


class TechnicalAccuracyChecker:
    """
    Validates technical accuracy of network design and troubleshooting samples
    using rule-based checks.
    """

    def check_protocol_usage(self, content: str) -> list[AccuracyCheckResult]:
        """
        Verify BGP/OSPF/EIGRP usage matches established guidelines.

        Args:
            content: Full assistant response text

        Returns:
            List of AccuracyCheckResult objects
        """
        results: list[AccuracyCheckResult] = []
        content_lower = content.lower()

        # Check VXLAN/EVPN only in DC context
        if any(kw in content_lower for kw in PROTOCOL_RULES["vxlan"]["keywords"]):
            has_dc_context = any(
                ctx in content_lower
                for ctx in PROTOCOL_RULES["vxlan"]["appropriate_for"]
            )
            results.append(AccuracyCheckResult(
                check_name="vxlan_context",
                passed=has_dc_context,
                severity="warning" if not has_dc_context else "info",
                message=(
                    "VXLAN/EVPN used appropriately in DC context"
                    if has_dc_context
                    else "VXLAN/EVPN mentioned without clear data center context"
                ),
            ))

        # Check EIGRP in multi-vendor context
        if any(kw in content_lower for kw in PROTOCOL_RULES["eigrp"]["keywords"]):
            has_multi_vendor = any(
                v in content_lower for v in ["juniper", "arista", "hp ", "extreme"]
            )
            if has_multi_vendor:
                results.append(AccuracyCheckResult(
                    check_name="eigrp_multi_vendor",
                    passed=False,
                    severity="warning",
                    message="EIGRP mentioned alongside non-Cisco vendors (EIGRP is Cisco proprietary)",
                ))

        return results

    def check_ip_addressing(self, content: str) -> list[AccuracyCheckResult]:
        """
        Validate IP addresses mentioned in the content.

        Args:
            content: Full response text

        Returns:
            List of AccuracyCheckResult objects
        """
        results: list[AccuracyCheckResult] = []

        # Find all IPv4 addresses
        ip_pattern = re.compile(
            r'\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b'
        )
        ips_found = ip_pattern.findall(content)

        for ip_str in ips_found:
            try:
                ip = ipaddress.IPv4Address(ip_str)
                # Check if public IPs used for internal design (possible error)
                if not ip.is_private and not ip.is_loopback:
                    # Skip if in examples/commands context (documentation addresses)
                    is_doc_addr = any(
                        str(ip).startswith(prefix)
                        for prefix in ["198.51.100.", "203.0.113.", "192.0.2."]
                    )
                    if not is_doc_addr:
                        results.append(AccuracyCheckResult(
                            check_name=f"public_ip_in_design_{ip_str}",
                            passed=False,
                            severity="warning",
                            message=f"Public IP {ip_str} used - verify if intentional (internet-facing)",
                            evidence=ip_str,
                        ))
            except ValueError:
                pass  # Invalid IP format - skip

        return results

    def check_topology_appropriateness(
        self,
        content: str,
        scenario: dict[str, Any] | None = None,
    ) -> list[AccuracyCheckResult]:
        """
        Verify topology recommendations match scenario requirements.

        Args:
            content: Full response text
            scenario: Optional scenario metadata for context

        Returns:
            List of AccuracyCheckResult objects
        """
        results: list[AccuracyCheckResult] = []
        content_lower = content.lower()

        # Check spine-leaf only in DC context
        if "spine-leaf" in content_lower or "spine leaf" in content_lower:
            rule = TOPOLOGY_RULES["spine_leaf"]
            has_dc_context = any(
                ctx in content_lower for ctx in rule["requires_context"]
            )
            results.append(AccuracyCheckResult(
                check_name="spine_leaf_appropriate",
                passed=has_dc_context,
                severity="error" if not has_dc_context else "info",
                message=(
                    rule["error_msg"] if not has_dc_context
                    else "Spine-leaf correctly used in DC context"
                ),
            ))

        # Check three-tier user count appropriateness
        if "three-tier" in content_lower or "three tier" in content_lower:
            if scenario and "user_count" in scenario:
                user_count = scenario["user_count"]
                min_users = TOPOLOGY_RULES["three_tier"]["appropriate_user_count_min"]
                is_appropriate = user_count >= min_users
                results.append(AccuracyCheckResult(
                    check_name="three_tier_scale",
                    passed=is_appropriate,
                    severity="warning" if not is_appropriate else "info",
                    message=(
                        f"Three-tier for {user_count} users is oversized "
                        f"(recommended: {min_users}+ users)"
                        if not is_appropriate
                        else f"Three-tier correctly sized for {user_count} users"
                    ),
                    evidence=str(user_count),
                ))

        # Check collapsed core user count limit
        if "collapsed core" in content_lower:
            if scenario and "user_count" in scenario:
                user_count = scenario["user_count"]
                max_users = TOPOLOGY_RULES["collapsed_core"]["max_user_count"]
                is_appropriate = user_count <= max_users
                results.append(AccuracyCheckResult(
                    check_name="collapsed_core_scale",
                    passed=is_appropriate,
                    severity="warning" if not is_appropriate else "info",
                    message=(
                        f"Collapsed core for {user_count} users exceeds recommended max "
                        f"({max_users} users)"
                        if not is_appropriate
                        else f"Collapsed core appropriate for {user_count} users"
                    ),
                    evidence=str(user_count),
                ))

        return results

    def check_compliance_completeness(
        self,
        content: str,
        required_compliance: list[str],
    ) -> list[AccuracyCheckResult]:
        """
        Verify all required compliance controls are addressed.

        Args:
            content: Full response text
            required_compliance: List of required compliance standards

        Returns:
            List of AccuracyCheckResult objects
        """
        results: list[AccuracyCheckResult] = []
        content_lower = content.lower()

        pci_required_controls = {
            "cde_segmentation": ["cde", "cardholder", "segmentation", "isolated vlan"],
            "encryption_tls": ["tls 1.", "tls1.", "encryption"],
            "firewall_rules": ["firewall", "acl", "access control"],
            "logging": ["log", "audit", "siem", "monitoring"],
        }

        hipaa_required_controls = {
            "phi_isolation": ["phi", "ehr", "emr", "hipaa vlan", "health data"],
            "encryption": ["encrypt", "tls", "aes"],
            "access_control": ["access control", "role-based", "least privilege", "mfa"],
        }

        if "PCI-DSS" in required_compliance:
            for control, keywords in pci_required_controls.items():
                has_control = any(kw in content_lower for kw in keywords)
                results.append(AccuracyCheckResult(
                    check_name=f"pci_{control}",
                    passed=has_control,
                    severity="error" if not has_control else "info",
                    message=(
                        f"PCI-DSS {control.replace('_', ' ')} not addressed"
                        if not has_control
                        else f"PCI-DSS {control.replace('_', ' ')} addressed"
                    ),
                ))

        if "HIPAA" in required_compliance:
            for control, keywords in hipaa_required_controls.items():
                has_control = any(kw in content_lower for kw in keywords)
                results.append(AccuracyCheckResult(
                    check_name=f"hipaa_{control}",
                    passed=has_control,
                    severity="error" if not has_control else "info",
                    message=(
                        f"HIPAA {control.replace('_', ' ')} not addressed"
                        if not has_control
                        else f"HIPAA {control.replace('_', ' ')} addressed"
                    ),
                ))

        return results

    def check_sample(self, sample: dict[str, Any], index: int = 0) -> SampleAccuracyReport:
        """Run all accuracy checks on a sample."""
        report = SampleAccuracyReport(sample_index=index)

        content = ""
        for msg in sample.get("messages", []):
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                break

        metadata = sample.get("metadata", {})
        scenario = metadata.get("scenario", {})
        compliance = scenario.get("compliance", [])

        # Run all checks
        for result in self.check_protocol_usage(content):
            report.add_check(result)

        for result in self.check_ip_addressing(content):
            report.add_check(result)

        for result in self.check_topology_appropriateness(content, scenario):
            report.add_check(result)

        if compliance:
            for result in self.check_compliance_completeness(content, compliance):
                report.add_check(result)

        report.calculate_score()
        return report

    def check_dataset(
        self,
        dataset_path: str | Path,
        sample_limit: int | None = None,
    ) -> dict[str, Any]:
        """
        Run accuracy checks on an entire dataset.

        Args:
            dataset_path: Path to JSONL file
            sample_limit: Max samples to check

        Returns:
            Aggregate statistics dict
        """
        path = Path(dataset_path)
        samples: list[dict] = []
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if sample_limit and i >= sample_limit:
                    break
                line = line.strip()
                if line:
                    try:
                        samples.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

        scores: list[float] = []
        critical_failures_total = 0

        for i, sample in enumerate(samples):
            report = self.check_sample(sample, i)
            scores.append(report.overall_score)
            critical_failures_total += report.critical_failures

        return {
            "total_samples": len(samples),
            "average_accuracy_score": sum(scores) / max(len(scores), 1),
            "total_critical_failures": critical_failures_total,
            "samples_with_critical_failures": sum(
                1 for s in scores if s < 60.0
            ),
        }


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="Check technical accuracy of training dataset"
    )
    parser.add_argument("dataset", help="Path to JSONL dataset file")
    parser.add_argument("--limit", type=int, default=None, help="Limit samples")
    args = parser.parse_args()

    checker = TechnicalAccuracyChecker()
    stats = checker.check_dataset(args.dataset, sample_limit=args.limit)

    console.print("\n[bold]Technical Accuracy Check Results[/bold]")
    for key, val in stats.items():
        if isinstance(val, float):
            console.print(f"  {key}: {val:.1f}%")
        else:
            console.print(f"  {key}: {val}")


if __name__ == "__main__":
    main()
