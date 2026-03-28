"""
Dataset Split Utility
Splits training data into train/validation/test sets.

Usage:
    # Default 80/10/10 split
    python data_generation/split.py \
        --input generated_data/training_data.jsonl

    # Custom split ratios
    python data_generation/split.py \
        --input generated_data/training_data.jsonl \
        --train 0.85 --val 0.10 --test 0.05

    # Stratified split by category
    python data_generation/split.py \
        --input generated_data/training_data.jsonl \
        --stratify
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load all samples from JSONL file."""
    samples: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Warning: Skipping invalid JSON at line {line_num}")
    return samples


def write_jsonl(samples: list[dict[str, Any]], path: Path) -> None:
    """Write samples to JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(samples):,} samples → {path}")


def split_random(
    samples: list[dict[str, Any]],
    train_ratio: float,
    val_ratio: float,
    seed: int = 42,
) -> tuple[list, list, list]:
    """Random split into train/val/test."""
    random.seed(seed)
    shuffled = samples.copy()
    random.shuffle(shuffled)

    n = len(shuffled)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    return shuffled[:train_end], shuffled[train_end:val_end], shuffled[val_end:]


def split_stratified(
    samples: list[dict[str, Any]],
    train_ratio: float,
    val_ratio: float,
    seed: int = 42,
) -> tuple[list, list, list]:
    """
    Stratified split that preserves category distribution.
    Uses 'type' or 'category' field from metadata.
    """
    random.seed(seed)

    # Group by category
    categories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        category = sample.get("metadata", {}).get("type", sample.get("metadata", {}).get("category", "unknown"))
        categories[category].append(sample)

    print(f"  Categories: {dict((k, len(v)) for k, v in categories.items())}")

    train_all: list[dict[str, Any]] = []
    val_all: list[dict[str, Any]] = []
    test_all: list[dict[str, Any]] = []

    for category, cat_samples in categories.items():
        random.shuffle(cat_samples)
        n = len(cat_samples)
        train_end = max(1, int(n * train_ratio))
        val_end = max(train_end + 1, int(n * (train_ratio + val_ratio)))

        train_all.extend(cat_samples[:train_end])
        val_all.extend(cat_samples[train_end:val_end])
        test_all.extend(cat_samples[val_end:])

    # Shuffle the combined sets
    random.shuffle(train_all)
    random.shuffle(val_all)
    random.shuffle(test_all)

    return train_all, val_all, test_all


def deduplicate(samples: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Remove duplicate samples based on prompt content."""
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    dupes = 0

    for sample in samples:
        messages = sample.get("messages", [])
        prompt = messages[0]["content"] if messages else str(sample)
        # Use first 200 chars as fingerprint
        fingerprint = prompt[:200].lower().strip()

        if fingerprint not in seen:
            seen.add(fingerprint)
            unique.append(sample)
        else:
            dupes += 1

    return unique, dupes


def validate_samples(samples: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Filter out samples that don't meet quality requirements."""
    valid: list[dict[str, Any]] = []
    invalid_count = 0

    for sample in samples:
        messages = sample.get("messages", [])

        # Must have user + assistant messages
        if len(messages) < 2:
            invalid_count += 1
            continue

        # Must have non-empty content
        if not messages[0].get("content") or not messages[-1].get("content"):
            invalid_count += 1
            continue

        # Assistant response should be substantial (>100 chars)
        assistant_content = messages[-1].get("content", "")
        if len(assistant_content) < 100:
            invalid_count += 1
            continue

        valid.append(sample)

    return valid, invalid_count


def compute_statistics(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute dataset statistics."""
    types: dict[str, int] = defaultdict(int)
    think_count = 0
    total_steps = 0
    total_len = 0

    for sample in samples:
        # Category
        category = sample.get("metadata", {}).get("type", "unknown")
        types[category] += 1

        # Think block presence
        messages = sample.get("messages", [])
        assistant = messages[-1].get("content", "") if messages else ""
        if "<think>" in assistant:
            think_count += 1
            # Count steps
            import re
            steps = len(re.findall(r"^Step\s+\d+:", assistant, re.MULTILINE))
            total_steps += steps

        total_len += sum(len(m.get("content", "")) for m in messages)

    n = len(samples) or 1
    return {
        "total": len(samples),
        "by_type": dict(types),
        "think_block_rate": think_count / n,
        "avg_response_chars": total_len / n,
        "avg_reasoning_steps": total_steps / max(think_count, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split training dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input",
        default=str(ROOT / "generated_data" / "training_data.jsonl"),
        help="Input JSONL file",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "generated_data"),
        help="Output directory for split files",
    )
    parser.add_argument("--train", type=float, default=0.80, help="Train ratio")
    parser.add_argument("--val", type=float, default=0.10, help="Validation ratio")
    parser.add_argument("--test", type=float, default=0.10, help="Test ratio")
    parser.add_argument("--stratify", action="store_true", help="Stratified split by category")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-dedup", action="store_true", help="Skip deduplication")
    parser.add_argument("--no-validate", action="store_true", help="Skip quality validation")
    args = parser.parse_args()

    # Validate ratios
    total_ratio = args.train + args.val + args.test
    if abs(total_ratio - 1.0) > 0.001:
        print(f"Error: train+val+test must sum to 1.0 (got {total_ratio:.3f})")
        sys.exit(1)

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        print("Generate first: python data_generation/dataset_generator.py --count 10000")
        sys.exit(1)

    print(f"Loading: {input_path}")
    samples = load_jsonl(input_path)
    print(f"Loaded: {len(samples):,} samples")

    # Deduplication
    if not args.no_dedup:
        samples, dupes = deduplicate(samples)
        print(f"Deduplication: removed {dupes} duplicates → {len(samples):,} unique")

    # Quality validation
    if not args.no_validate:
        samples, invalid = validate_samples(samples)
        print(f"Validation: removed {invalid} invalid → {len(samples):,} valid")

    # Print statistics
    stats = compute_statistics(samples)
    print(f"\nDataset statistics:")
    print(f"  Total samples: {stats['total']:,}")
    print(f"  By type: {stats['by_type']}")
    print(f"  Think block rate: {stats['think_block_rate']:.1%}")
    print(f"  Avg response chars: {stats['avg_response_chars']:.0f}")
    print(f"  Avg reasoning steps: {stats['avg_reasoning_steps']:.1f}")

    # Split
    print(f"\nSplitting ({args.train:.0%} / {args.val:.0%} / {args.test:.0%})...")
    if args.stratify:
        print("Using stratified split")
        train, val, test = split_stratified(samples, args.train, args.val, args.seed)
    else:
        train, val, test = split_random(samples, args.train, args.val, args.seed)

    print(f"\nSplit results:")
    print(f"  Train: {len(train):,}")
    print(f"  Val:   {len(val):,}")
    print(f"  Test:  {len(test):,}")

    # Write output files
    print("\nWriting output files:")
    write_jsonl(train, output_dir / "training_data.jsonl")
    write_jsonl(val, output_dir / "validation_data.jsonl")
    write_jsonl(test, output_dir / "test_data.jsonl")

    print("\nDone! Files ready for training:")
    print(f"  python fine_tuning/train.py")


if __name__ == "__main__":
    main()
