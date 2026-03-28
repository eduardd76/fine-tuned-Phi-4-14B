"""
Dataset Generator for Phi-4 Virtual Network Architect Fine-Tuning

Generates 10,000+ training samples grounded in network design knowledge.
Each sample has structured chain-of-thought reasoning in <think> tags.

Usage:
    export OPENAI_API_KEY=sk-...
    python dataset_generator.py --count 10000
    python dataset_generator.py --count 100 --output_dir ./test_output
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

from openai import OpenAI
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from knowledge_extraction.extract_from_notebooklm import NotebookLMKnowledgeExtractor
from data_generation.reasoning_chain_builder import ReasoningChainBuilder

console = Console()

KNOWLEDGE_DIR = Path(__file__).parent.parent / "knowledge_extraction"
OUTPUT_DIR = Path(__file__).parent.parent / "generated_data"

DESIGN_INDUSTRIES = [
    "financial_services", "healthcare", "retail", "manufacturing",
    "education", "government", "technology", "hospitality",
    "transportation", "energy_utilities",
]

COMPLIANCE_COMBOS = [
    [], ["PCI-DSS"], ["HIPAA"], ["SOX"], ["PCI-DSS", "SOX"],
    ["HIPAA", "SOX"], ["PCI-DSS", "HIPAA", "SOX"], ["NIST-CSF"],
    ["ISO-27001"], ["FedRAMP"],
]

TROUBLESHOOTING_SCENARIOS = [
    {"protocol": "bgp", "issue_type": "neighbor_down", "symptoms": ["BGP neighbor in Active state", "Routes missing from table"]},
    {"protocol": "bgp", "issue_type": "route_missing", "symptoms": ["Expected prefix not in BGP table", "Traffic blackholing"]},
    {"protocol": "bgp", "issue_type": "convergence_slow", "symptoms": ["BGP takes 45+ seconds to converge", "Traffic disruption during failover"]},
    {"protocol": "ospf", "issue_type": "neighbor_exstart", "symptoms": ["OSPF neighbor stuck in EXSTART state", "DBD exchange failing"]},
    {"protocol": "ospf", "issue_type": "lsa_flooding", "symptoms": ["High CPU on routers", "Frequent LSA retransmissions"]},
    {"protocol": "ospf", "issue_type": "neighbor_down", "symptoms": ["OSPF adjacency dropped", "Dead timer expired"]},
    {"protocol": "interface", "issue_type": "packet_loss", "symptoms": ["Intermittent packet loss", "High error counters"]},
    {"protocol": "interface", "issue_type": "flapping", "symptoms": ["Interface cycling up/down", "Syslog flood of link state changes"]},
    {"protocol": "interface", "issue_type": "crc_errors", "symptoms": ["CRC errors incrementing", "Degraded throughput"]},
    {"protocol": "qos", "issue_type": "buffer_drops", "symptoms": ["Voice quality degraded", "Queue drops on WAN interface"]},
    {"protocol": "qos", "issue_type": "misclassification", "symptoms": ["Business apps not getting priority bandwidth", "Wrong DSCP markings"]},
    {"protocol": "connectivity", "issue_type": "layer3", "symptoms": ["Cannot reach destination", "Traceroute loops detected"]},
]


class DatasetGenerator:
    """
    Generates fine-tuning samples for the virtual network architect model.

    Each sample contains:
    - A realistic network design or troubleshooting question
    - A structured response with <think> chain-of-thought
    - Metadata with sources, patterns, and compliance references
    """

    def __init__(self, openai_api_key: str | None = None) -> None:
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not set. Export it: export OPENAI_API_KEY=sk-..."
            )
        self.client = OpenAI(api_key=api_key)
        self.extractor = NotebookLMKnowledgeExtractor()
        self.knowledge = self._load_all_knowledge()
        self.chain_builder = ReasoningChainBuilder(self.knowledge)
        self._stats = {"design": 0, "troubleshooting": 0, "errors": 0}

    def _load_all_knowledge(self) -> dict[str, Any]:
        """Load all knowledge bases into a single dict."""
        knowledge: dict[str, Any] = {}
        for category in ["design_patterns", "troubleshooting_trees",
                          "compliance_requirements", "vendor_specifics",
                          "configuration_templates", "cost_benchmarks"]:
            path = KNOWLEDGE_DIR / f"{category}.json"
            if path.exists():
                with open(path, encoding="utf-8") as f:
                    knowledge[category] = json.load(f)
        return knowledge

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def generate_dataset(
        self,
        count: int,
        output_dir: Path | None = None,
        design_ratio: float = 0.70,
    ) -> dict[str, int]:
        """
        Generate a complete dataset with train/val/test splits.

        Args:
            count: Total number of samples to generate
            output_dir: Directory for output files
            design_ratio: Fraction of design vs troubleshooting samples

        Returns:
            Dict with counts: total, train, val, test, errors
        """
        out_dir = output_dir or OUTPUT_DIR
        out_dir.mkdir(parents=True, exist_ok=True)

        design_count = int(count * design_ratio)
        trouble_count = count - design_count

        console.print(f"\n[bold]Dataset Generation Plan[/bold]")
        console.print(f"Total samples: {count}")
        console.print(f"  Design: {design_count} ({design_ratio*100:.0f}%)")
        console.print(f"  Troubleshooting: {trouble_count} ({(1-design_ratio)*100:.0f}%)")
        console.print(f"Output: {out_dir}\n")

        all_samples: list[dict[str, Any]] = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            design_task = progress.add_task(
                "[cyan]Generating design samples...", total=design_count
            )
            for i in range(design_count):
                scenario = self._random_design_scenario()
                sample = self._generate_design_sample(scenario)
                if sample:
                    all_samples.append(sample)
                    self._stats["design"] += 1
                else:
                    self._stats["errors"] += 1
                progress.advance(design_task)

                # Rate limiting
                if (i + 1) % 20 == 0:
                    time.sleep(1)

            trouble_task = progress.add_task(
                "[yellow]Generating troubleshooting samples...", total=trouble_count
            )
            for i in range(trouble_count):
                scenario = random.choice(TROUBLESHOOTING_SCENARIOS)
                sample = self._generate_troubleshooting_sample(scenario)
                if sample:
                    all_samples.append(sample)
                    self._stats["troubleshooting"] += 1
                else:
                    self._stats["errors"] += 1
                progress.advance(trouble_task)

                if (i + 1) % 20 == 0:
                    time.sleep(1)

        # Shuffle before splitting
        random.shuffle(all_samples)

        # 80/10/10 split
        total = len(all_samples)
        train_end = int(total * 0.80)
        val_end = int(total * 0.90)

        splits = {
            "training_data": all_samples[:train_end],
            "validation_data": all_samples[train_end:val_end],
            "test_data": all_samples[val_end:],
        }

        for name, data in splits.items():
            path = out_dir / f"{name}.jsonl"
            with open(path, "w", encoding="utf-8") as f:
                for sample in data:
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            console.print(f"[green]Wrote[/green] {len(data)} samples → {path.name}")

        result = {
            "total": total,
            "train": len(splits["training_data"]),
            "val": len(splits["validation_data"]),
            "test": len(splits["test_data"]),
            "errors": self._stats["errors"],
        }

        console.print(f"\n[bold green]Generation complete![/bold green]")
        console.print(f"Train: {result['train']}, Val: {result['val']}, Test: {result['test']}")
        console.print(f"Errors: {result['errors']}")
        return result

    # ------------------------------------------------------------------ #
    # Sample generation                                                    #
    # ------------------------------------------------------------------ #

    def _generate_design_sample(
        self, scenario: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Generate a single design scenario training sample."""
        try:
            # Build reasoning chain locally
            chain = self.chain_builder.build_design_chain(scenario, self.knowledge)
            think_block = chain.to_think_block()

            # Build the user question
            user_question = self._build_design_question(scenario)

            # Call GPT-4o-mini to generate the actual answer
            system_prompt = self._build_design_system_prompt()
            assistant_content = self._call_llm(
                system_prompt=system_prompt,
                user_message=user_question,
                think_block=think_block,
            )

            if not assistant_content:
                return None

            return {
                "messages": [
                    {"role": "user", "content": user_question},
                    {"role": "assistant", "content": f"{think_block}\n\n{assistant_content}"},
                ],
                "metadata": {
                    "type": "design",
                    "scenario": scenario,
                    "sources_used": [
                        "Enterprise Network Design - Topology Selection Criteria",
                        "PCI-DSS v4.0.1 Network Requirements",
                        "Enterprise Network TCO Model",
                        "Vendor Comparison Guide",
                    ],
                    "design_patterns": [
                        self.chain_builder._select_topology(scenario).replace("_", " "),
                    ],
                    "compliance_requirements": scenario.get("compliance", []),
                    "reasoning_steps": chain.step_count(),
                },
            }
        except Exception as exc:
            console.print(f"[red]Design sample error: {exc}[/red]")
            return None

    def _generate_troubleshooting_sample(
        self, scenario: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Generate a single troubleshooting training sample."""
        try:
            chain = self.chain_builder.build_troubleshooting_chain(scenario, self.knowledge)
            think_block = chain.to_think_block()

            user_question = self._build_troubleshooting_question(scenario)
            system_prompt = self._build_troubleshooting_system_prompt()

            assistant_content = self._call_llm(
                system_prompt=system_prompt,
                user_message=user_question,
                think_block=think_block,
            )

            if not assistant_content:
                return None

            return {
                "messages": [
                    {"role": "user", "content": user_question},
                    {"role": "assistant", "content": f"{think_block}\n\n{assistant_content}"},
                ],
                "metadata": {
                    "type": "troubleshooting",
                    "scenario": scenario,
                    "sources_used": [
                        f"{scenario['protocol'].upper()} Troubleshooting Decision Tree",
                        "OSI Model Troubleshooting Methodology",
                        "Network Diagnostic Command Reference",
                    ],
                    "design_patterns": [],
                    "compliance_requirements": [],
                    "reasoning_steps": chain.step_count(),
                },
            }
        except Exception as exc:
            console.print(f"[red]Troubleshooting sample error: {exc}[/red]")
            return None

    def _call_llm(
        self,
        system_prompt: str,
        user_message: str,
        think_block: str,
    ) -> str | None:
        """Call GPT-4o-mini with retry logic."""
        prompt = (
            f"The reasoning chain is already provided. Using this reasoning, "
            f"write the final structured response.\n\n"
            f"Reasoning chain:\n{think_block}\n\n"
            f"Question: {user_message}"
        )
        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=2000,
                    temperature=0.7,
                )
                return response.choices[0].message.content
            except Exception as exc:
                if attempt == 2:
                    console.print(f"[red]LLM call failed after 3 attempts: {exc}[/red]")
                    return None
                wait = 2 ** attempt
                time.sleep(wait)
        return None

    # ------------------------------------------------------------------ #
    # Scenario builders                                                    #
    # ------------------------------------------------------------------ #

    def _random_design_scenario(self) -> dict[str, Any]:
        """Generate a random but realistic network design scenario."""
        scale = random.choice(["small", "medium", "large", "enterprise"])
        scale_ranges = {
            "small": (50, 500),
            "medium": (500, 5000),
            "large": (5000, 50000),
            "enterprise": (50000, 200000),
        }
        lo, hi = scale_ranges[scale]
        user_count = random.randint(lo, hi)

        site_count = random.choice([1, 1, 2, 3, 5, 10, 25, 50, 100])
        industry = random.choice(DESIGN_INDUSTRIES)
        compliance = random.choice(COMPLIANCE_COMBOS)
        uptime = random.choice(["99.9%", "99.9%", "99.99%", "99.999%"])
        budget = random.choice(["constrained", "moderate", "flexible", "unlimited"])
        is_datacenter = random.random() < 0.15

        # Industry-compliance correlations
        if industry == "financial_services" and not compliance:
            compliance = ["PCI-DSS", "SOX"]
        elif industry == "healthcare" and not compliance:
            compliance = ["HIPAA"]

        return {
            "user_count": user_count,
            "site_count": site_count,
            "industry": industry,
            "compliance": compliance,
            "uptime_requirement": uptime,
            "budget": budget,
            "is_datacenter": is_datacenter,
            "cloud_heavy": random.random() < 0.3,
            "scale": scale,
        }

    def _build_design_question(self, scenario: dict[str, Any]) -> str:
        """Build a natural language design question from a scenario."""
        compliance_str = (
            f" We must comply with {' and '.join(scenario['compliance'])}."
            if scenario["compliance"] else ""
        )
        budget_str = {
            "constrained": "limited budget",
            "moderate": "moderate budget",
            "flexible": "flexible budget",
            "unlimited": "budget is not a primary concern",
        }.get(scenario["budget"], "moderate budget")

        industry_str = scenario["industry"].replace("_", " ")

        templates = [
            (
                f"Design a network for a {industry_str} company with "
                f"{scenario['user_count']:,} users across "
                f"{scenario['site_count']} {'sites' if scenario['site_count'] > 1 else 'site'}."
                f"{compliance_str} We need {scenario['uptime_requirement']} uptime and have a {budget_str}."
            ),
            (
                f"I'm building network infrastructure for a {industry_str} organization. "
                f"We have {scenario['user_count']:,} employees "
                f"{'distributed across ' + str(scenario['site_count']) + ' locations' if scenario['site_count'] > 1 else 'at a single location'}."
                f"{compliance_str} Reliability target: {scenario['uptime_requirement']}. Budget: {budget_str}. "
                f"What architecture do you recommend?"
            ),
            (
                f"Recommend a complete network architecture for a {industry_str} enterprise: "
                f"{scenario['user_count']:,} users, {scenario['site_count']} sites."
                f"{compliance_str} Required uptime: {scenario['uptime_requirement']}. "
                f"Budget constraints: {budget_str}. Include topology, WAN design, security, "
                f"hardware recommendations, and cost estimates."
            ),
        ]
        return random.choice(templates)

    def _build_troubleshooting_question(self, scenario: dict[str, Any]) -> str:
        """Build a natural language troubleshooting question."""
        symptoms_str = "; ".join(scenario["symptoms"])
        protocol = scenario["protocol"].upper()
        issue = scenario["issue_type"].replace("_", " ")

        templates = [
            (
                f"We're experiencing {issue} on our {protocol} network. "
                f"Symptoms: {symptoms_str}. Walk me through the troubleshooting methodology."
            ),
            (
                f"Our network team is seeing: {symptoms_str}. This appears to be a "
                f"{protocol} {issue} issue. What is the systematic approach to diagnose and resolve this?"
            ),
            (
                f"Help me troubleshoot a {protocol} problem. "
                f"We're observing: {symptoms_str}. "
                f"Provide step-by-step diagnostic commands and expected outputs."
            ),
        ]
        return random.choice(templates)

    def _build_design_system_prompt(self) -> str:
        """System prompt for design scenario generation."""
        return """You are a Virtual Network Architect with CCDE-level expertise.
You provide detailed, technically accurate network design recommendations grounded in
industry-proven methodologies.

Your responses must:
1. Follow the reasoning chain provided
2. Include specific hardware recommendations with model numbers
3. Reference relevant standards (PCI-DSS v4.0.1, HIPAA, NIST CSF 2.0 etc.)
4. Provide realistic cost estimates using industry benchmarks
5. Include implementation timeline and phasing
6. Use proper technical terminology (OSPF, BGP, HSRP, QoS, VXLAN, etc.)
7. Structure response with clear sections: topology, WAN, security, hardware, costs, roadmap

Format: Markdown with headers for each major section."""

    def _build_troubleshooting_system_prompt(self) -> str:
        """System prompt for troubleshooting scenario generation."""
        return """You are an expert network engineer with CCIE-level troubleshooting expertise.
You follow systematic diagnostic methodologies grounded in proven frameworks.

Your responses must:
1. Follow the reasoning chain and diagnostic steps provided
2. Include specific CLI commands with expected outputs
3. Explain the reason behind each diagnostic step
4. Provide the root cause analysis with evidence
5. Give clear remediation steps
6. Include verification commands to confirm resolution
7. Mention monitoring steps to prevent recurrence

Format: Numbered troubleshooting steps with commands and expected outputs."""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate training dataset for Phi-4 Virtual Network Architect",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--count", type=int, default=10000,
        help="Total number of training samples to generate"
    )
    parser.add_argument(
        "--output_dir", type=Path, default=OUTPUT_DIR,
        help="Output directory for generated JSONL files"
    )
    parser.add_argument(
        "--design_ratio", type=float, default=0.70,
        help="Fraction of design vs troubleshooting samples (0.0 - 1.0)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--api_key", type=str, default=None,
        help="OpenAI API key (overrides OPENAI_API_KEY env var)"
    )
    args = parser.parse_args()

    random.seed(args.seed)

    console.print("[bold blue]Phi-4 Virtual Network Architect Dataset Generator[/bold blue]")
    console.print(f"Model: gpt-4o-mini | Count: {args.count} | Seed: {args.seed}\n")

    generator = DatasetGenerator(openai_api_key=args.api_key)
    result = generator.generate_dataset(
        count=args.count,
        output_dir=args.output_dir,
        design_ratio=args.design_ratio,
    )

    console.print(f"\n[bold]Final Statistics[/bold]")
    for key, val in result.items():
        console.print(f"  {key}: {val}")


if __name__ == "__main__":
    main()
