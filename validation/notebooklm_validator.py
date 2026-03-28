"""
NotebookLM Validator

Validates that generated training samples correctly use the knowledge base,
have proper reasoning chains, and meet quality standards.

Usage:
    python notebooklm_validator.py generated_data/training_data.jsonl
    python notebooklm_validator.py generated_data/training_data.jsonl --strict
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

console = Console()

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class ValidationResult:
    """Result of validating a single sample."""
    passed: bool
    checks: dict[str, bool] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    score: float = 0.0

    def add_check(self, name: str, passed: bool, message: str = "") -> None:
        self.checks[name] = passed
        if not passed and message:
            self.errors.append(f"{name}: {message}")

    def calculate_score(self) -> None:
        if not self.checks:
            self.score = 0.0
            return
        self.score = sum(self.checks.values()) / len(self.checks) * 100
        self.passed = self.score >= 70.0


@dataclass
class DatasetValidationReport:
    """Summary report for a full dataset validation."""
    total_samples: int = 0
    passed_samples: int = 0
    failed_samples: int = 0
    average_score: float = 0.0
    check_pass_rates: dict[str, float] = field(default_factory=dict)
    common_errors: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


KNOWN_SOURCES = [
    "Enterprise Network Design",
    "Network Design",
    "PCI-DSS",
    "HIPAA",
    "NIST",
    "ISO 27001",
    "SOX",
    "Cisco",
    "Juniper",
    "Arista",
    "BGP",
    "OSPF",
    "SD-WAN",
    "Zero Trust",
    "Topology",
    "Compliance",
    "TCO Model",
    "CapEx",
    "RFC",
    "CCDE",
    "Chapter",
    "Section",
    "Requirement",
    "Guide",
]

COMPLIANCE_VERSIONS = {
    "PCI-DSS": ["4.0.1", "4.0", "3.2.1"],
    "HIPAA": ["Security Rule", "Privacy Rule", "164.312"],
    "SOX": ["Section 302", "Section 404"],
    "NIST": ["CSF 2.0", "CSF 1.1", "800-53"],
}


class NotebookLMValidator:
    """
    Validates generated training samples for quality and knowledge grounding.

    Checks:
    - Presence and structure of <think> tags
    - Minimum reasoning steps (5+)
    - Source attribution in reasoning
    - Technical accuracy of claims
    - Compliance version specificity
    - JSON structure validity
    """

    def __init__(self, strict: bool = False) -> None:
        self.strict = strict
        self.min_reasoning_steps = 5
        self.min_source_references = 3 if strict else 2
        self.min_response_length = 500 if strict else 200

    def validate_reasoning_chain(self, sample: dict[str, Any]) -> ValidationResult:
        """
        Validate the <think>...</think> reasoning chain.

        Args:
            sample: Training sample dict with 'messages' key

        Returns:
            ValidationResult with check details
        """
        result = ValidationResult(passed=False)
        messages = sample.get("messages", [])

        if len(messages) < 2:
            result.add_check("has_messages", False, "Sample must have at least 2 messages")
            result.calculate_score()
            return result

        assistant_msg = None
        for msg in messages:
            if msg.get("role") == "assistant":
                assistant_msg = msg.get("content", "")
                break

        if not assistant_msg:
            result.add_check("has_assistant_message", False, "No assistant message found")
            result.calculate_score()
            return result

        # Check 1: Has <think> tags
        has_think = "<think>" in assistant_msg and "</think>" in assistant_msg
        result.add_check(
            "has_think_tags", has_think,
            "Missing <think>...</think> tags in assistant response"
        )

        # Check 2: Extract think block content
        think_match = re.search(r"<think>(.*?)</think>", assistant_msg, re.DOTALL)
        think_content = think_match.group(1) if think_match else ""

        # Check 3: Minimum reasoning steps
        step_matches = re.findall(r"Step\s+\d+:", think_content, re.IGNORECASE)
        step_count = len(step_matches)
        has_min_steps = step_count >= self.min_reasoning_steps
        result.add_check(
            "min_reasoning_steps", has_min_steps,
            f"Only {step_count} steps found (minimum {self.min_reasoning_steps})"
        )

        # Check 4: Source references in think block
        source_refs = sum(
            1 for src in KNOWN_SOURCES
            if src.lower() in think_content.lower()
        )
        has_sources = source_refs >= self.min_source_references
        result.add_check(
            "has_source_references", has_sources,
            f"Only {source_refs} source references found (minimum {self.min_source_references})"
        )

        # Check 5: Content after </think> (actual answer)
        after_think = assistant_msg.split("</think>")[-1].strip()
        has_answer = len(after_think) >= self.min_response_length
        result.add_check(
            "has_answer_content", has_answer,
            f"Answer section too short: {len(after_think)} chars (min {self.min_response_length})"
        )

        result.calculate_score()
        return result

    def validate_technical_accuracy(self, sample: dict[str, Any]) -> ValidationResult:
        """
        Validate technical accuracy using rule-based checks.

        Args:
            sample: Training sample dict

        Returns:
            ValidationResult with technical accuracy checks
        """
        result = ValidationResult(passed=False)
        messages = sample.get("messages", [])

        # Get full assistant content
        content = ""
        for msg in messages:
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                break

        content_lower = content.lower()

        # Check: If spine-leaf mentioned, DC context required
        if "spine-leaf" in content_lower or "spine leaf" in content_lower:
            has_dc_context = any(
                word in content_lower
                for word in ["data center", "datacenter", "server", "rack", "pod"]
            )
            result.add_check(
                "spine_leaf_context", has_dc_context,
                "Spine-leaf topology mentioned without data center context"
            )

        # Check: If BGP mentioned for small network, flag warning
        metadata = sample.get("metadata", {})
        scenario = metadata.get("scenario", {})
        user_count = scenario.get("user_count", 0)

        if "ebgp" in content_lower and user_count and user_count < 500:
            result.warnings.append(
                "eBGP used for small network (<500 users) - verify this is intentional"
            )

        # Check: Compliance versions when compliance mentioned
        content_lower_full = content.lower()
        for standard, versions in COMPLIANCE_VERSIONS.items():
            if standard.lower() in content_lower_full:
                has_version = any(v.lower() in content_lower_full for v in versions)
                result.add_check(
                    f"compliance_version_{standard}", has_version,
                    f"{standard} mentioned without version number"
                )

        # Check: TLS version when encryption mentioned
        if "tls" in content_lower or "ssl" in content_lower:
            has_tls_version = re.search(r"tls\s*1\.[2-3]", content_lower)
            result.add_check(
                "tls_version_specified",
                bool(has_tls_version),
                "TLS mentioned without version (should specify TLS 1.2 or 1.3)"
            )

        # Check: User question present
        has_user_question = any(
            msg.get("role") == "user" for msg in messages
        )
        result.add_check("has_user_question", has_user_question, "No user question found")

        # If no specific checks were added, pass by default
        if not result.checks:
            result.checks["basic_structure"] = True

        result.calculate_score()
        return result

    def validate_source_attribution(self, sample: dict[str, Any]) -> ValidationResult:
        """
        Validate that technical claims can be traced to sources.

        Args:
            sample: Training sample dict

        Returns:
            ValidationResult checking source attribution
        """
        result = ValidationResult(passed=False)
        messages = sample.get("messages", [])

        content = ""
        for msg in messages:
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                break

        # Check think block for source references
        think_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
        think_content = think_match.group(1) if think_match else ""

        # Count "Source:" or "Reference:" patterns
        source_patterns = re.findall(
            r"(?:Source|Reference|Ref|Per|From|Chapter|Section)[:.]?\s*[^\n]+",
            think_content, re.IGNORECASE
        )
        has_explicit_sources = len(source_patterns) >= 2
        result.add_check(
            "explicit_source_citations", has_explicit_sources,
            f"Only {len(source_patterns)} explicit source citations in think block"
        )

        # Check metadata for sources_used
        metadata = sample.get("metadata", {})
        sources_used = metadata.get("sources_used", [])
        has_metadata_sources = len(sources_used) >= 2
        result.add_check(
            "metadata_sources_populated", has_metadata_sources,
            "metadata.sources_used has fewer than 2 entries"
        )

        # Check reasoning_steps count in metadata
        reasoning_steps = metadata.get("reasoning_steps", 0)
        has_steps = reasoning_steps >= self.min_reasoning_steps
        result.add_check(
            "metadata_reasoning_steps", has_steps,
            f"metadata.reasoning_steps = {reasoning_steps} (min {self.min_reasoning_steps})"
        )

        result.calculate_score()
        return result

    def validate_sample(self, sample: dict[str, Any]) -> ValidationResult:
        """Run all validations on a single sample."""
        r1 = self.validate_reasoning_chain(sample)
        r2 = self.validate_technical_accuracy(sample)
        r3 = self.validate_source_attribution(sample)

        # Merge results
        combined = ValidationResult(passed=False)
        combined.checks.update(r1.checks)
        combined.checks.update(r2.checks)
        combined.checks.update(r3.checks)
        combined.errors.extend(r1.errors + r2.errors + r3.errors)
        combined.warnings.extend(r1.warnings + r2.warnings + r3.warnings)
        combined.calculate_score()
        return combined

    def validate_dataset(
        self,
        dataset_path: str | Path,
        sample_limit: int | None = None,
    ) -> DatasetValidationReport:
        """
        Validate an entire dataset file.

        Args:
            dataset_path: Path to JSONL file
            sample_limit: Max samples to validate (None = all)

        Returns:
            DatasetValidationReport with aggregate statistics
        """
        path = Path(dataset_path)
        if not path.exists():
            console.print(f"[red]File not found: {path}[/red]")
            return DatasetValidationReport()

        samples: list[dict[str, Any]] = []
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if sample_limit and i >= sample_limit:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError as e:
                    console.print(f"[red]JSON parse error on line {i+1}: {e}[/red]")

        console.print(f"\nValidating {len(samples)} samples from {path.name}...")

        report = DatasetValidationReport()
        report.total_samples = len(samples)
        all_scores: list[float] = []
        check_counts: dict[str, list[bool]] = {}
        error_freq: dict[str, int] = {}

        from rich.progress import Progress as RichProgress
        with RichProgress(console=console) as progress:
            task = progress.add_task("Validating...", total=len(samples))
            for sample in samples:
                result = self.validate_sample(sample)
                if result.passed:
                    report.passed_samples += 1
                else:
                    report.failed_samples += 1

                all_scores.append(result.score)

                for check, passed in result.checks.items():
                    check_counts.setdefault(check, []).append(passed)

                for error in result.errors:
                    key = error.split(":")[0]
                    error_freq[key] = error_freq.get(key, 0) + 1

                progress.advance(task)

        report.average_score = sum(all_scores) / max(len(all_scores), 1)
        report.check_pass_rates = {
            k: sum(v) / len(v) * 100
            for k, v in check_counts.items()
        }
        # Top 5 most common errors
        sorted_errors = sorted(error_freq.items(), key=lambda x: x[1], reverse=True)
        report.common_errors = [f"{k}: {v} occurrences" for k, v in sorted_errors[:5]]

        # Recommendations
        for check, rate in report.check_pass_rates.items():
            if rate < 80.0:
                report.recommendations.append(
                    f"Improve '{check}': currently {rate:.1f}% pass rate"
                )

        self._print_report(report, path.name)
        return report

    def _print_report(self, report: DatasetValidationReport, filename: str) -> None:
        """Print a formatted validation report."""
        console.print(f"\n[bold]Validation Report: {filename}[/bold]")
        console.print(f"Total: {report.total_samples} | "
                      f"[green]Passed: {report.passed_samples}[/green] | "
                      f"[red]Failed: {report.failed_samples}[/red] | "
                      f"Avg Score: {report.average_score:.1f}%\n")

        if report.check_pass_rates:
            table = Table(title="Check Pass Rates")
            table.add_column("Check", style="cyan")
            table.add_column("Pass Rate", justify="right")
            table.add_column("Status")

            for check, rate in sorted(report.check_pass_rates.items()):
                status = "[green]GOOD[/green]" if rate >= 80 else "[red]NEEDS WORK[/red]"
                table.add_row(check, f"{rate:.1f}%", status)
            console.print(table)

        if report.common_errors:
            console.print("\n[bold]Most Common Issues:[/bold]")
            for err in report.common_errors:
                console.print(f"  [red]•[/red] {err}")

        if report.recommendations:
            console.print("\n[bold]Recommendations:[/bold]")
            for rec in report.recommendations:
                console.print(f"  [yellow]→[/yellow] {rec}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate training dataset quality",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("dataset", help="Path to JSONL dataset file")
    parser.add_argument(
        "--strict", action="store_true",
        help="Enable strict validation (higher thresholds)"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit number of samples to validate"
    )
    args = parser.parse_args()

    validator = NotebookLMValidator(strict=args.strict)
    report = validator.validate_dataset(args.dataset, sample_limit=args.limit)

    # Exit code based on pass rate
    pass_rate = report.passed_samples / max(report.total_samples, 1)
    sys.exit(0 if pass_rate >= 0.80 else 1)


if __name__ == "__main__":
    main()
