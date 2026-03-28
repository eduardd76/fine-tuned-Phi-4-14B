"""
Batch Processing Script
Process multiple network design queries through Phi-4 in bulk.

Useful for:
- Generating design documents for multiple sites
- Batch evaluation of network scenarios
- Pre-computing responses for a FAQ database

Usage:
    # Process JSONL file of queries
    python deployment/batch_process.py \
        --model models/phi4-network-architect \
        --input queries.jsonl \
        --output results.jsonl

    # Process CSV file
    python deployment/batch_process.py \
        --model models/phi4-gptq \
        --input queries.csv \
        --output results.jsonl \
        --format csv

    # Test with built-in sample queries
    python deployment/batch_process.py \
        --model models/phi4-network-architect \
        --sample
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

SAMPLE_QUERIES = [
    {
        "id": "sample_001",
        "category": "campus_design",
        "prompt": "Design a three-tier campus network for a university with 15,000 students across 8 buildings. Include redundancy, wireless coverage, and 10G uplinks to the core.",
    },
    {
        "id": "sample_002",
        "category": "data_center",
        "prompt": "Design a spine-leaf data center for a financial services company with 2,000 servers, 10ms latency SLA, and PCI-DSS 4.0.1 compliance.",
    },
    {
        "id": "sample_003",
        "category": "wan_design",
        "prompt": "Design WAN architecture for a retail chain with 500 stores. Each store needs 100 Mbps primary and 50 Mbps failover connectivity. Cost optimization is critical.",
    },
    {
        "id": "sample_004",
        "category": "troubleshooting",
        "prompt": "BGP neighbor between PE and CE router is stuck in Active state. The peer IP is a loopback address. Provide step-by-step troubleshooting methodology.",
    },
    {
        "id": "sample_005",
        "category": "compliance",
        "prompt": "Design network segmentation for a hospital with 800 beds. Must comply with HIPAA and include medical device isolation, EHR access controls, and guest WiFi separation.",
    },
    {
        "id": "sample_006",
        "category": "sd_wan",
        "prompt": "Should we migrate from DMVPN Phase 3 to SD-WAN? We have 200 branch offices. Compare the solutions considering cost, management overhead, and application performance.",
    },
    {
        "id": "sample_007",
        "category": "security",
        "prompt": "Design zone-based security architecture for a manufacturing plant with OT/IT convergence. Include ICS/SCADA isolation, remote access, and DMZ design.",
    },
    {
        "id": "sample_008",
        "category": "mpls",
        "prompt": "We need to connect 5 global data centers across 3 service providers. Design an inter-AS MPLS VPN solution with traffic engineering and failover.",
    },
    {
        "id": "sample_009",
        "category": "troubleshooting",
        "prompt": "OSPF adjacency is stuck in EXSTART state between two routers on an Ethernet segment. One is a Cisco 9300, the other is an Arista 7050. Diagnose and resolve.",
    },
    {
        "id": "sample_010",
        "category": "qos",
        "prompt": "Design end-to-end QoS policy for a unified communications deployment with 500 IP phones, video conferencing, and critical business applications on a 1 Gbps WAN link.",
    },
]


def load_queries_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load queries from JSONL file."""
    queries: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            # Support multiple formats
            if "prompt" not in item and "messages" in item:
                item["prompt"] = item["messages"][0]["content"]
            if "id" not in item:
                item["id"] = f"q_{i:04d}"
            queries.append(item)
    return queries


def load_queries_csv(path: Path) -> list[dict[str, Any]]:
    """Load queries from CSV file (columns: id, prompt, category)."""
    queries: list[dict[str, Any]] = []
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, 1):
            queries.append({
                "id": row.get("id", f"q_{i:04d}"),
                "prompt": row.get("prompt", row.get("query", row.get("question", ""))),
                "category": row.get("category", "general"),
            })
    return queries


def process_batch(
    queries: list[dict[str, Any]],
    model_path: str,
    output_path: Path,
    max_new_tokens: int = 2048,
    temperature: float = 0.7,
) -> dict[str, Any]:
    """
    Process all queries through the model.

    Returns:
        Summary statistics
    """
    sys.path.insert(0, str(Path(__file__).parent.parent / "deployment"))
    from inference import HFInferenceBackend, infer

    print(f"Loading model: {model_path}")
    backend = HFInferenceBackend(model_path, load_in_4bit=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    stats = {
        "total": len(queries),
        "completed": 0,
        "failed": 0,
        "avg_latency_ms": 0.0,
        "think_block_rate": 0.0,
    }

    latencies: list[float] = []
    think_count = 0

    with open(output_path, "w", encoding="utf-8") as f_out:
        for i, query in enumerate(queries):
            query_id = query.get("id", f"q_{i+1:04d}")
            prompt = query.get("prompt", "")
            category = query.get("category", "general")

            if not prompt:
                stats["failed"] += 1
                continue

            print(f"  [{i+1}/{len(queries)}] {query_id}: {prompt[:60]}...", end="\r")

            try:
                result = infer(prompt, backend, max_new_tokens, temperature)

                output_item = {
                    "id": query_id,
                    "category": category,
                    "prompt": prompt,
                    "reasoning": result["reasoning"],
                    "answer": result["answer"],
                    "sources": result["sources"],
                    "latency_ms": result["latency_ms"],
                    "has_think_block": result["has_think_block"],
                    "reasoning_steps": result["reasoning_steps"],
                }

                f_out.write(json.dumps(output_item, ensure_ascii=False) + "\n")
                results.append(output_item)

                latencies.append(result["latency_ms"])
                if result["has_think_block"]:
                    think_count += 1
                stats["completed"] += 1

            except Exception as e:
                print(f"\n  ERROR on {query_id}: {e}")
                stats["failed"] += 1
                # Write error record
                error_item = {
                    "id": query_id,
                    "category": category,
                    "prompt": prompt,
                    "error": str(e),
                    "answer": "",
                    "reasoning": "",
                }
                f_out.write(json.dumps(error_item, ensure_ascii=False) + "\n")

    print()  # Clear progress line

    if latencies:
        stats["avg_latency_ms"] = sum(latencies) / len(latencies)
        stats["max_latency_ms"] = max(latencies)
        stats["min_latency_ms"] = min(latencies)
    if stats["completed"] > 0:
        stats["think_block_rate"] = think_count / stats["completed"]

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch Process Network Design Queries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model", required=False, default=str(ROOT / "models" / "phi4-network-architect"))
    parser.add_argument("--input", default=None, help="Input queries file (JSONL or CSV)")
    parser.add_argument("--output", default="batch_results.jsonl")
    parser.add_argument("--format", choices=["jsonl", "csv"], default="jsonl")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--sample", action="store_true", help="Run on built-in sample queries")
    args = parser.parse_args()

    if args.sample:
        queries = SAMPLE_QUERIES
        print(f"Using {len(queries)} built-in sample queries")
    elif args.input:
        input_path = Path(args.input)
        if args.format == "csv" or input_path.suffix == ".csv":
            queries = load_queries_csv(input_path)
        else:
            queries = load_queries_jsonl(input_path)
        print(f"Loaded {len(queries)} queries from {input_path}")
    else:
        print("Error: Provide --input or --sample")
        sys.exit(1)

    output_path = Path(args.output)
    stats = process_batch(queries, args.model, output_path, args.max_tokens, args.temperature)

    print(f"\n{'='*50}")
    print("Batch Processing Complete")
    print(f"  Total queries:    {stats['total']}")
    print(f"  Completed:        {stats['completed']}")
    print(f"  Failed:           {stats['failed']}")
    print(f"  Avg latency:      {stats['avg_latency_ms']:.0f}ms")
    print(f"  Think block rate: {stats['think_block_rate']:.1%}")
    print(f"  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
