"""
run_pipeline.py — Orchestrator for all 3 dataset generators.
Called by run_all_stages.sh: python run_pipeline.py --type [sft|grpo|agentic|all]
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime


def run_sft(output_path: str) -> None:
    """Run SFT dataset generator. Uses template generator (no API key needed)
    or OpenAI-backed generator if OPENAI_API_KEY is set."""
    sys.path.insert(0, str(Path(__file__).parent))
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if api_key and api_key not in ("", "your-key-here"):
        from dataset_generator import DatasetGenerator
        generator = DatasetGenerator(openai_api_key=api_key)
        generator.generate_dataset(count=9000, output_dir=Path(output_path))
    else:
        print("  No OPENAI_API_KEY — using template generator (no API calls)")
        from template_dataset_generator import generate_sft_dataset
        generate_sft_dataset(output_path, count=9000)
    print(f"✓ SFT dataset generation complete → {output_path}")


def run_grpo(output_path: str, target: int = 800) -> None:
    """Run GRPO dataset generator (template-based, no API key needed)."""
    sys.path.insert(0, str(Path(__file__).parent))
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if api_key and api_key not in ("", "your-key-here"):
        from grpo_generator import generate_grpo_dataset
        generate_grpo_dataset(output_path, target)
    else:
        print("  No OPENAI_API_KEY — using template GRPO generator")
        from template_dataset_generator import generate_grpo_dataset_template
        generate_grpo_dataset_template(output_path, target)


def run_agentic(output_path: str, target: int = 2000) -> None:
    """Run agentic trajectory generator."""
    sys.path.insert(0, str(Path(__file__).parent))
    from agentic_generator import generate_agentic_dataset
    generate_agentic_dataset(output_path, target)


def validate_output(output_path: str) -> dict:
    """Validate all generated datasets."""
    output_dir = Path(output_path)
    results = {}

    for fname in ["sft_train.jsonl", "sft_val.jsonl", "grpo_train.jsonl", "agentic_train.jsonl"]:
        fpath = output_dir / fname
        if fpath.exists():
            with open(fpath) as f:
                count = sum(1 for line in f if line.strip())
            results[fname] = {"exists": True, "count": count}
        else:
            results[fname] = {"exists": False, "count": 0}

    return results


def main():
    parser = argparse.ArgumentParser(description="Phi-4 Dataset Generation Pipeline")
    parser.add_argument("--type", choices=["sft", "grpo", "agentic", "all"], required=True)
    parser.add_argument("--output", default="/data/datasets/")
    parser.add_argument("--grpo-target", type=int, default=800)
    parser.add_argument("--agentic-target", type=int, default=2000)
    parser.add_argument("--smoke-test", action="store_true",
                        help="Generate 10 samples per type for testing")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable not set")
        sys.exit(1)

    output_path = args.output
    Path(output_path).mkdir(parents=True, exist_ok=True)

    if args.smoke_test:
        args.grpo_target = 10
        args.agentic_target = 10
        print("Smoke test mode — generating 10 samples per type")

    start = datetime.now()
    print(f"\n{'='*50}")
    print(f"Dataset Generation Pipeline — {args.type.upper()}")
    print(f"Output: {output_path}")
    print(f"{'='*50}\n")

    try:
        if args.type in ("sft", "all"):
            print("[1/3] Generating SFT dataset...")
            run_sft(output_path)

        if args.type in ("grpo", "all"):
            print("[2/3] Generating GRPO dataset...")
            run_grpo(output_path, args.grpo_target)

        if args.type in ("agentic", "all"):
            print("[3/3] Generating Agentic dataset...")
            run_agentic(output_path, args.agentic_target)

    except KeyboardInterrupt:
        print("\nInterrupted — partial datasets may exist in", output_path)
        sys.exit(1)

    elapsed = (datetime.now() - start).total_seconds()
    results = validate_output(output_path)

    print(f"\n{'='*50}")
    print(f"Pipeline complete in {elapsed:.0f}s")
    print(f"{'='*50}")
    for fname, info in results.items():
        status = "OK" if info["exists"] else "MISSING"
        print(f"  [{status}] {fname}: {info['count']} samples")

    manifest = {
        "generated_at": datetime.now().isoformat(),
        "elapsed_seconds": elapsed,
        "output_path": output_path,
        "datasets": results,
    }
    with open(Path(output_path) / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest written: {output_path}/manifest.json")


if __name__ == "__main__":
    main()
