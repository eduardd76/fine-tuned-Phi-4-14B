"""
stage_tracker.py — Track evaluation scores across all 3 training stages.
Saves results to stages_progress.json for comparison and course documentation.
"""
from __future__ import annotations
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

STAGES_FILE = Path(__file__).parent.parent / "stages_progress.json"
TEST_CASES_FILE = Path(__file__).parent / "test_cases.jsonl"


def load_stages_progress() -> dict:
    if STAGES_FILE.exists():
        with open(STAGES_FILE) as f:
            return json.load(f)
    return {
        "baseline_gpt4o_mini": {
            "overall_score": 0.44, "keyword_score": 0.13, "think_rate": 0.96,
            "pass_count": 1, "total_cases": 25,
            "evaluated_at": "2026-03-28T13:22:00Z",
            "notes": "GPT-4o-mini baseline before fine-tuning",
        },
        "stage1_sft": None,
        "stage2_grpo": None,
        "stage3_agentic": None,
    }


def run_evaluation_local(model_path: str, test_cases_path: str) -> dict:
    print(f"  Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    test_cases = []
    with open(test_cases_path) as f:
        for line in f:
            if line.strip():
                test_cases.append(json.loads(line))

    results = []
    for i, tc in enumerate(test_cases):
        start = time.time()
        messages = [
            {"role": "system", "content": "You are a CCDE-certified network architect."},
            {"role": "user", "content": tc["question"]},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=800, temperature=0.7,
                                     do_sample=True, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        latency = time.time() - start

        expected_facts = tc.get("expected_key_facts", [])
        sources_required = tc.get("sources_required", [])
        kw_score = sum(1 for f in expected_facts if f.lower() in response.lower()) / max(len(expected_facts), 1)
        src_score = sum(1 for s in sources_required if s.lower() in response.lower()) / max(len(sources_required), 1) if sources_required else 1.0
        has_think = "<think>" in response
        confidence = min(0.9, len(response.split()) / 300 + (0.2 if has_think else 0))
        overall = 0.50 * kw_score + 0.20 * src_score + 0.20 * confidence + (0.10 if has_think else 0)
        pass_fail = "PASS" if overall >= 0.70 else ("WARN" if overall >= 0.45 else "FAIL")

        results.append({
            "test_id": tc["id"], "category": tc["category"],
            "overall_score": round(overall, 3), "keyword_score": round(kw_score, 3),
            "has_think": has_think, "latency_s": round(latency, 2), "pass_fail": pass_fail,
            "keywords_missing": [f for f in expected_facts if f.lower() not in response.lower()],
        })
        print(f"  [{i+1:2d}/{len(test_cases)}] {tc['id']}: {pass_fail} ({overall:.2f})")

    n = len(results)
    return {
        "overall_score": round(sum(r["overall_score"] for r in results) / n, 3),
        "keyword_score": round(sum(r["keyword_score"] for r in results) / n, 3),
        "think_rate": round(sum(1 for r in results if r["has_think"]) / n, 3),
        "pass_count": sum(1 for r in results if r["pass_fail"] == "PASS"),
        "total_cases": n,
        "per_test": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, required=True, choices=[1, 2, 3])
    parser.add_argument("--model", required=True)
    parser.add_argument("--test-cases", default=str(TEST_CASES_FILE))
    args = parser.parse_args()

    stage_key = f"stage{args.stage}_{'sft' if args.stage == 1 else 'grpo' if args.stage == 2 else 'agentic'}"
    scores = run_evaluation_local(args.model, args.test_cases)
    progress = load_stages_progress()
    progress[stage_key] = {**scores, "model_path": args.model,
                           "evaluated_at": datetime.utcnow().isoformat() + "Z"}

    with open(STAGES_FILE, "w") as f:
        json.dump(progress, f, indent=2)

    baseline = progress["baseline_gpt4o_mini"]
    print(f"\n{'='*55}")
    print(f"{'Metric':<25} {'Baseline':>10} {'Stage '+str(args.stage):>12} {'Delta':>8}")
    print("-" * 55)
    for metric in ["overall_score", "keyword_score", "think_rate"]:
        b, s = baseline.get(metric, 0), scores.get(metric, 0)
        sign = "+" if s - b >= 0 else ""
        print(f"  {metric:<23} {b:>10.3f} {s:>12.3f} {sign+f'{s-b:.3f}':>8}")
    print(f"\n  PASS: {baseline['pass_count']}/25 → {scores['pass_count']}/25")
    print(f"Results saved: {STAGES_FILE}")


if __name__ == "__main__":
    main()
