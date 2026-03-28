"""
agentic_eval.py — Evaluate 4 agentic behaviours after Stage 3 training.
Behaviours: tool_selection, react_loop, clarification, escalation.
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

AGENTIC_TEST_CASES_FILE = Path(__file__).parent / "agentic_test_cases.jsonl"


def load_model(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, messages: list, max_new_tokens: int = 600) -> str:
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                  temperature=0.7, do_sample=True,
                                  pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)


def score_tool_selection(response: str, expected_tool: str) -> dict:
    has_call = "<tool_call>" in response
    m = re.search(r'<tool_call>(.*?)</tool_call>', response, re.DOTALL)
    correct = False
    if m:
        try:
            correct = json.loads(m.group(1)).get("tool", "") == expected_tool
        except Exception:
            pass
    score = 0.4 * has_call + 0.4 * correct + 0.2 * ("<think>" in response)
    return {"passed": score >= 0.8, "score": score, "has_call": has_call, "correct_tool": correct}


def score_clarification(response: str) -> dict:
    q_count = sum(1 for p in [r"\?", r"could you (clarify|specify)", r"(how many|what|which)",
                               r"before (designing|recommending)", r"I need (more|additional)"]
                  if re.search(p, response, re.IGNORECASE))
    premature = bool(re.search(r"I recommend (spine-leaf|hub-and-spoke|full mesh)", response[:200], re.IGNORECASE))
    score = min(1.0, q_count * 0.25) * (0.8 if not premature else 0.4) + 0.2 * ("<think>" in response)
    return {"passed": q_count >= 3 and not premature, "score": score, "questions": q_count}


def score_escalation(response: str, expected_agent: str) -> dict:
    correct = any(re.search(p, response, re.IGNORECASE) for p in [
        rf"escalat.*{expected_agent}", rf"{expected_agent}.*agent", rf"A2A.*escalat"])
    reason = bool(re.search(r"because|reason|since|requires?", response, re.IGNORECASE))
    score = 0.5 * correct + 0.3 * reason + 0.2 * ("<think>" in response)
    return {"passed": correct, "score": score, "correct_agent": correct}


def score_react_loop(response: str) -> dict:
    has_think = "<think>" in response
    has_call = "<tool_call>" in response
    order_ok = response.find("<think>") < response.find("<tool_call>") if (has_think and has_call) else False
    has_answer = bool(re.search(r"(recommend|suggest|root cause|the (issue|fix|solution))", response, re.IGNORECASE))
    score = 0.25 * has_think + 0.35 * has_call + 0.20 * order_ok + 0.20 * has_answer
    return {"passed": score >= 0.75, "score": score, "has_think": has_think, "has_call": has_call}


SCORERS = {
    "tool_selection": score_tool_selection,
    "clarification": score_clarification,
    "escalation": score_escalation,
    "react_loop": score_react_loop,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--test-cases", default=str(AGENTIC_TEST_CASES_FILE))
    parser.add_argument("--output", default="agentic_eval_results.json")
    args = parser.parse_args()

    model, tokenizer = load_model(args.model)
    test_cases = [json.loads(l) for l in open(args.test_cases) if l.strip()]

    results = []
    for tc in test_cases:
        response = generate(model, tokenizer, tc["messages"])
        behaviour = tc["behaviour"]
        scorer = SCORERS.get(behaviour)
        if not scorer:
            continue

        if behaviour == "tool_selection":
            r = scorer(response, tc.get("expected_tool", ""))
        elif behaviour == "escalation":
            r = scorer(response, tc.get("expected_agent", ""))
        else:
            r = scorer(response)

        results.append({"test_id": tc["id"], "behaviour": behaviour, **r})
        print(f"  {'✓' if r['passed'] else '✗'} {tc['id']} ({behaviour}): {r['score']:.2f}")

    overall = sum(1 for r in results if r["passed"]) / len(results) if results else 0
    print(f"\nOverall pass rate: {overall:.1%}")

    beh_summary = {}
    for beh in SCORERS:
        br = [r for r in results if r["behaviour"] == beh]
        if br:
            beh_summary[beh] = {
                "pass_rate": sum(1 for r in br if r["passed"]) / len(br),
                "avg_score": sum(r["score"] for r in br) / len(br),
            }
            print(f"  {beh}: {beh_summary[beh]['pass_rate']:.1%} pass")

    with open(args.output, "w") as f:
        json.dump({"overall_pass_rate": overall, "behaviours": beh_summary, "per_test": results}, f, indent=2)
    print(f"Results: {args.output}")


if __name__ == "__main__":
    main()
