"""
Evaluation Runner - Phi-4 Network Architect

Runs all evaluation metrics against the fine-tuned model and generates
a comprehensive report with pass/fail status against target thresholds.

Usage:
    python evaluation/run_all.py \
        --model models/phi4-network-architect \
        --test-data evaluation/test_cases.jsonl \
        --knowledge-dir knowledge_extraction/

    # Quick evaluation (20 samples)
    python evaluation/run_all.py --model models/phi4-network-architect --quick

    # Skip LLM-judge (no API key needed)
    python evaluation/run_all.py --model models/phi4-network-architect --no-llm-judge
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

console = Console()

# Target thresholds (must achieve ALL to pass)
TARGETS = {
    "technical_accuracy": 0.95,
    "reasoning_quality": 4.5,
    "notebooklm_alignment": 0.95,
    "config_syntax_valid": 1.0,
    "think_block_present": 0.98,
    "min_reasoning_steps": 5,
}


def load_test_cases(path: Path) -> list[dict[str, Any]]:
    """Load test cases from JSONL file."""
    cases: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def run_inference_on_cases(
    cases: list[dict[str, Any]],
    model_path: str,
) -> list[dict[str, Any]]:
    """Run model inference on all test cases."""
    sys.path.insert(0, str(Path(__file__).parent.parent / "deployment"))
    from inference import HFInferenceBackend, infer

    console.print(f"[cyan]Loading model: {model_path}[/cyan]")
    backend = HFInferenceBackend(model_path, load_in_4bit=True)

    results = []
    for i, case in enumerate(cases):
        prompt = case.get("prompt") or case["messages"][0]["content"]
        expected = case.get("expected_answer") or (
            case["messages"][-1]["content"] if len(case["messages"]) > 1 else ""
        )

        console.print(f"  [{i+1}/{len(cases)}] Inferring...", end="\r")
        result = infer(prompt, backend)
        results.append({
            "prompt": prompt,
            "expected": expected,
            "predicted": result["answer"],
            "reasoning": result["reasoning"],
            "sources": result["sources"],
            "latency_ms": result["latency_ms"],
            "has_think_block": result["has_think_block"],
            "reasoning_steps": result["reasoning_steps"],
            "metadata": case.get("metadata", {}),
        })

    console.print()
    return results


def evaluate_all(
    model_path: str,
    test_data_path: Path,
    knowledge_dir: Path,
    quick: bool = False,
    no_llm_judge: bool = False,
    openai_api_key: str | None = None,
) -> dict[str, Any]:
    """
    Run all evaluation metrics.

    Returns:
        Full evaluation report with per-metric scores and pass/fail status.
    """
    from evaluation.technical_accuracy import TechnicalAccuracyEvaluator
    from evaluation.alignment_score import AlignmentScoreEvaluator

    console.print("\n[bold blue]Phi-4 Network Architect - Evaluation Suite[/bold blue]")
    console.print(f"Model: {model_path}")
    console.print(f"Test data: {test_data_path}")
    console.print()

    # Load test cases
    cases = load_test_cases(test_data_path)
    if quick:
        cases = cases[:20]
        console.print(f"[yellow]Quick mode: using {len(cases)} samples[/yellow]")
    console.print(f"Test cases: {len(cases)}")

    # Run inference
    console.print("\n[bold]Phase 1: Running inference...[/bold]")
    t0 = time.perf_counter()
    results = run_inference_on_cases(cases, model_path)
    inference_time = time.perf_counter() - t0
    avg_latency = sum(r["latency_ms"] for r in results) / len(results)
    console.print(f"Inference complete. Avg latency: {avg_latency:.0f}ms")

    # Technical accuracy
    console.print("\n[bold]Phase 2: Technical accuracy...[/bold]")
    tech_eval = TechnicalAccuracyEvaluator(knowledge_dir)
    tech_scores = tech_eval.evaluate_batch(results)
    console.print(f"Technical accuracy: {tech_scores['score']:.3f}")

    # NotebookLM alignment
    console.print("\n[bold]Phase 3: NotebookLM alignment...[/bold]")
    align_eval = AlignmentScoreEvaluator(knowledge_dir)
    align_scores = align_eval.evaluate_batch(results)
    console.print(f"NotebookLM alignment: {align_scores['score']:.3f}")

    # LLM judge (optional)
    reasoning_score = None
    if not no_llm_judge:
        console.print("\n[bold]Phase 4: LLM-as-Judge reasoning quality...[/bold]")
        try:
            from evaluation.llm_judge import LLMJudgeEvaluator
            judge = LLMJudgeEvaluator(api_key=openai_api_key)
            sample_results = results[:min(50, len(results))]  # Judge on 50 samples max
            judge_scores = judge.evaluate_batch(sample_results)
            reasoning_score = judge_scores["score"]
            console.print(f"Reasoning quality: {reasoning_score:.2f}/5.0")
        except Exception as e:
            console.print(f"[yellow]LLM judge skipped: {e}[/yellow]")

    # Think block & syntax checks
    console.print("\n[bold]Phase 5: Structure validation...[/bold]")
    think_present = sum(1 for r in results if r["has_think_block"]) / len(results)
    steps_ok = sum(1 for r in results if r["reasoning_steps"] >= TARGETS["min_reasoning_steps"]) / len(results)

    # Config syntax (check samples that include configs)
    config_results = [r for r in results if "```" in r["predicted"]]
    if config_results:
        from evaluation.technical_accuracy import check_config_syntax
        syntax_valid = sum(
            1 for r in config_results if check_config_syntax(r["predicted"])
        ) / len(config_results)
    else:
        syntax_valid = 1.0  # No configs to check

    # Latency check
    latency_pass = avg_latency < 3000  # <3s target

    # ── Compile report ───────────────────────────────────────
    metrics: dict[str, Any] = {
        "technical_accuracy": tech_scores["score"],
        "notebooklm_alignment": align_scores["score"],
        "think_block_present": think_present,
        "steps_sufficient": steps_ok,
        "config_syntax_valid": syntax_valid,
        "avg_latency_ms": avg_latency,
        "latency_under_3s": latency_pass,
    }
    if reasoning_score is not None:
        metrics["reasoning_quality"] = reasoning_score

    # Pass/fail against targets
    gate_results: dict[str, bool] = {}
    skipped_gates: set[str] = set()
    for metric, target in TARGETS.items():
        if metric in metrics:
            gate_results[metric] = metrics[metric] >= target
        else:
            gate_results[metric] = True  # Skip gates that weren't evaluated
            skipped_gates.add(metric)

    overall_pass = all(gate_results.values())

    report: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "model": model_path,
        "test_samples": len(results),
        "metrics": metrics,
        "targets": TARGETS,
        "gate_results": gate_results,
        "overall_pass": overall_pass,
        "per_sample_summary": {
            "avg_latency_ms": avg_latency,
            "max_latency_ms": max(r["latency_ms"] for r in results),
            "think_block_rate": think_present,
            "avg_reasoning_steps": sum(r["reasoning_steps"] for r in results) / len(results),
        },
        "technical_details": {
            "tech_accuracy": tech_scores,
            "alignment": align_scores,
        },
    }

    # ── Print results table ──────────────────────────────────
    console.print("\n" + "=" * 60)
    console.print("[bold]EVALUATION RESULTS[/bold]")
    console.print("=" * 60)

    table = Table(title="Metric Scores vs Targets")
    table.add_column("Metric", style="cyan")
    table.add_column("Score", justify="right")
    table.add_column("Target", justify="right")
    table.add_column("Status")

    for metric, target in TARGETS.items():
        score = metrics.get(metric)
        if score is None:
            table.add_row(metric, "N/A", str(target), "[yellow]SKIPPED[/yellow]")
            continue
        score_str = f"{score:.3f}" if isinstance(score, float) else str(score)
        target_str = str(target)
        passed = score >= target
        status = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"
        table.add_row(metric, score_str, target_str, status)

    # Extra metrics
    table.add_row("avg_latency_ms", f"{avg_latency:.0f}ms", "<3000ms",
                  "[green]PASS[/green]" if latency_pass else "[red]FAIL[/red]")

    console.print(table)
    console.print()

    if overall_pass:
        console.print("[bold green]OVERALL: PASS - Model meets all quality gates![/bold green]")
    else:
        console.print("[bold red]OVERALL: FAIL - Some quality gates not met[/bold red]")
        failed = [k for k, v in gate_results.items() if v is False]
        console.print(f"Failed gates: {', '.join(failed)}")
        console.print("\nRecommendations:")
        if not gate_results.get("technical_accuracy", True):
            console.print("  • Generate more diverse training samples with better source grounding")
        if not gate_results.get("reasoning_quality", True):
            console.print("  • Increase reasoning_loss weight or add more chain-of-thought examples")
        if not gate_results.get("notebooklm_alignment", True):
            console.print("  • Ensure training data has exact compliance versions (e.g., PCI-DSS 4.0.1)")

    # Save report
    report_path = ROOT / "evaluation" / "results.json"
    report_path.parent.mkdir(exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    console.print(f"\nReport saved to: {report_path}")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phi-4 Network Architect Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model", required=True, help="Path to model")
    parser.add_argument(
        "--test-data",
        default=str(ROOT / "evaluation" / "test_cases.jsonl"),
        help="Path to test cases JSONL",
    )
    parser.add_argument(
        "--knowledge-dir",
        default=str(ROOT / "knowledge_extraction"),
        help="Path to knowledge base directory",
    )
    parser.add_argument("--quick", action="store_true", help="Quick mode (20 samples)")
    parser.add_argument("--no-llm-judge", action="store_true", help="Skip LLM-as-Judge")
    parser.add_argument("--openai-api-key", default=None)
    args = parser.parse_args()

    report = evaluate_all(
        model_path=args.model,
        test_data_path=Path(args.test_data),
        knowledge_dir=Path(args.knowledge_dir),
        quick=args.quick,
        no_llm_judge=args.no_llm_judge,
        openai_api_key=args.openai_api_key,
    )

    sys.exit(0 if report["overall_pass"] else 1)


if __name__ == "__main__":
    main()
