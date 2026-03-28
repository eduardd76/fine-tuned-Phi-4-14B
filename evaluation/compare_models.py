"""
compare_models.py — Multi-model comparison, generates COMPARISON_REPORT.md.
"""
from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path

STAGES_FILE = Path(__file__).parent.parent / "stages_progress.json"
COMPARISON_REPORT = Path(__file__).parent.parent / "COMPARISON_REPORT.md"

BASELINE = {
    "name": "GPT-4o-mini (baseline)",
    "overall_score": 0.44, "keyword_score": 0.13, "think_rate": 0.96,
    "pass_count": 1, "total_cases": 25,
    "per_category": {
        "compliance": 0.61, "topology": 0.49, "vxlan": 0.49, "routing": 0.45,
        "ha": 0.44, "ospf": 0.44, "bgp": 0.46, "qos": 0.39, "wan": 0.41,
        "security": 0.44, "mpls": 0.36, "sdwan": 0.33, "design_methodology": 0.35,
    },
}


def load_stages() -> dict:
    if STAGES_FILE.exists():
        with open(STAGES_FILE) as f:
            return json.load(f)
    return {}


def generate_report(stages: dict) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Phi-4 Network Architect — Model Comparison Report",
        "", f"**Generated:** {now}",
        "**Test suite:** 25 CCDE-level scenarios (`evaluation/test_cases.jsonl`)",
        "", "---", "", "## Score Progression",
        "",
        "| Model | Overall | Keywords | `<think>` | PASS/25 |",
        "|-------|--------:|---------:|----------:|--------:|",
    ]

    b = BASELINE
    lines.append(f"| {b['name']} | {b['overall_score']:.0%} | {b['keyword_score']:.0%} | {b['think_rate']:.0%} | {b['pass_count']} |")

    for key, label in [("stage1_sft", "Stage 1 SFT"), ("stage2_grpo", "Stage 2 GRPO"), ("stage3_agentic", "Stage 3 Agentic")]:
        if stages.get(key):
            s = stages[key]
            lines.append(f"| {label} | {s['overall_score']:.0%} | {s['keyword_score']:.0%} | {s['think_rate']:.0%} | {s['pass_count']} |")
        else:
            lines.append(f"| {label} | — | — | — | — |")

    lines.extend(["| **Target** | **≥70%** | **≥65%** | **≥95%** | **≥18** |", ""])

    lines.extend(["---", "", "## Per-Category Breakdown", "",
                  "| Category | Baseline | Stage 1 | Stage 2 | Stage 3 | Target |",
                  "|----------|--------:|--------:|--------:|--------:|-------:|"])

    for cat in sorted(BASELINE["per_category"]):
        b_score = BASELINE["per_category"][cat]
        def get_cat(key):
            if stages.get(key) and stages[key].get("per_test"):
                cr = [r for r in stages[key]["per_test"] if r["category"] == cat]
                return f"{sum(r['overall_score'] for r in cr)/len(cr):.2f}" if cr else "—"
            return "—"
        lines.append(f"| {cat} | {b_score:.2f} | {get_cat('stage1_sft')} | {get_cat('stage2_grpo')} | {get_cat('stage3_agentic')} | 0.70 |")

    lines.extend(["", "---", "", "*Regenerate: `python evaluation/compare_models.py`*"])
    return "\n".join(lines)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(COMPARISON_REPORT))
    args = parser.parse_args()

    stages = load_stages()
    report = generate_report(stages)
    Path(args.output).write_text(report)
    print(f"✓ Comparison report: {args.output}")

    for key in ["baseline_gpt4o_mini", "stage1_sft", "stage2_grpo", "stage3_agentic"]:
        if stages.get(key):
            s = stages[key]
            print(f"  {key}: overall={s.get('overall_score',0):.3f} pass={s.get('pass_count',0)}/25")
        else:
            print(f"  {key}: not yet evaluated")


if __name__ == "__main__":
    main()
