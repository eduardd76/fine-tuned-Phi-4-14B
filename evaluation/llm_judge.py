"""
LLM-as-Judge Evaluator

Uses GPT-4o or Claude to evaluate reasoning quality on a 1-5 scale.
Evaluates: chain-of-thought structure, technical depth, source attribution,
compliance specificity, and design completeness.

Scoring rubric (1-5):
    5 - Perfect: Deep reasoning, 8+ steps, all sources cited, exact compliance versions
    4 - Good: Solid reasoning, 5-7 steps, most sources cited
    3 - Adequate: Basic reasoning, 3-4 steps, some sources
    2 - Weak: Minimal reasoning, <3 steps, few sources
    1 - Poor: No structured reasoning, unsourced claims
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

JUDGE_SYSTEM_PROMPT = """You are a senior network architect evaluating AI-generated network design responses.
Rate the response quality on a scale of 1-5 based on the rubric provided.
Return ONLY a JSON object with: {"score": <1-5>, "rationale": "<brief reason>"}"""

JUDGE_USER_TEMPLATE = """## Evaluation Task

Rate this network architect AI response on a 1-5 scale.

**Question:** {question}

**Response:** {response}

**Scoring Rubric:**
- 5: Perfect - Structured <think> reasoning (8+ steps), every claim cited to a source, exact compliance version numbers (e.g., "PCI-DSS 4.0.1 §1.2"), vendor-validated configs, cost estimates grounded in data
- 4: Good - 5-7 reasoning steps, most claims sourced, compliance frameworks mentioned with versions
- 3: Adequate - 3-4 reasoning steps, general sources mentioned, compliance frameworks named
- 2: Weak - Minimal reasoning, few or no sources, generic compliance statements
- 1: Poor - No structured reasoning, unsupported claims, missing source attribution

**Required format:** JSON only: {{"score": <1-5>, "rationale": "<25 words max>"}}"""


class LLMJudgeEvaluator:
    """Evaluate reasoning quality using an LLM judge."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        provider: str = "openai",
    ) -> None:
        self.model = model
        self.provider = provider

        if provider == "openai":
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        elif provider == "anthropic":
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def judge_single(self, question: str, response: str) -> dict[str, Any]:
        """
        Judge a single question/response pair.

        Returns:
            {"score": 1-5, "rationale": "...", "raw": "..."}
        """
        user_msg = JUDGE_USER_TEMPLATE.format(
            question=question[:500],  # Truncate long questions
            response=response[:3000],  # Truncate long responses
        )

        raw = ""
        try:
            if self.provider == "openai":
                from openai import OpenAI as _OpenAI
                assert isinstance(self.client, _OpenAI)
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=0.1,
                    max_tokens=200,
                    response_format={"type": "json_object"},
                )
                raw = resp.choices[0].message.content or ""
            elif self.provider == "anthropic":
                import anthropic as _anthropic
                assert isinstance(self.client, _anthropic.Anthropic)
                resp = self.client.messages.create(
                    model=self.model,
                    system=JUDGE_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_msg}],
                    max_tokens=200,
                )
                raw = "".join(
                    getattr(block, "text", "")
                    for block in resp.content
                )
            else:
                return {"score": 3.0, "rationale": "Unknown provider", "raw": ""}

            parsed = json.loads(raw)
            score = float(parsed.get("score", 3))
            score = max(1.0, min(5.0, score))  # Clamp to 1-5

            return {
                "score": score,
                "rationale": parsed.get("rationale", ""),
                "raw": raw,
            }

        except json.JSONDecodeError:
            # Try to extract score with regex
            score_match = re.search(r'"score"\s*:\s*([1-5](?:\.\d)?)', raw)
            if score_match:
                return {
                    "score": float(score_match.group(1)),
                    "rationale": "Parsed from malformed JSON",
                    "raw": raw,
                }
            return {"score": 3.0, "rationale": "Parse error", "raw": raw}

        except Exception as e:
            return {"score": 3.0, "rationale": f"Error: {str(e)[:50]}", "raw": ""}

    def evaluate_batch(
        self,
        results: list[dict[str, Any]],
        rate_limit_pause: float = 0.5,
    ) -> dict[str, Any]:
        """
        Evaluate a batch of inference results.

        Args:
            results: List of inference result dicts with 'prompt' and 'predicted'
            rate_limit_pause: Seconds between API calls

        Returns:
            Batch evaluation summary with average score
        """
        scores: list[float] = []
        judgements: list[dict[str, Any]] = []

        for i, result in enumerate(results):
            question = result.get("prompt", "")
            # Evaluate the full response (reasoning + answer)
            response = result.get("reasoning", "") + "\n\n" + result.get("predicted", "")

            judgement = self.judge_single(question, response)
            scores.append(judgement["score"])
            judgements.append(judgement)

            if (i + 1) % 10 == 0:
                avg_so_far = sum(scores) / len(scores)
                print(f"  Judged {i+1}/{len(results)}, avg={avg_so_far:.2f}", end="\r")

            time.sleep(rate_limit_pause)

        print()
        avg_score = sum(scores) / len(scores) if scores else 0.0

        # Score distribution
        distribution = {str(i): sum(1 for s in scores if int(s) == i) for i in range(1, 6)}

        return {
            "score": avg_score,
            "num_evaluated": len(scores),
            "score_distribution": distribution,
            "min_score": min(scores) if scores else 0.0,
            "max_score": max(scores) if scores else 0.0,
            "samples_above_4": sum(1 for s in scores if s >= 4.0),
            "samples_below_3": sum(1 for s in scores if s < 3.0),
            "judgements": judgements,
        }


class LocalReasoningScorer:
    """
    Score reasoning quality without an LLM API.
    Uses heuristics based on structure and content.
    """

    def score(self, response: str) -> float:
        """
        Score a response on a 1-5 scale using structural heuristics.

        Criteria:
        - Has <think> block: +1 point
        - Reasoning steps (5+): +1 point
        - Source citations (5+): +1 point
        - Compliance version numbers: +1 point
        - Config examples: +0.5 point
        """
        score = 1.0

        # Has <think> block
        if re.search(r"<think>", response, re.IGNORECASE):
            score += 1.0

        # Reasoning steps
        steps = len(re.findall(r"^Step\s+\d+:", response, re.MULTILINE))
        if steps >= 8:
            score += 1.0
        elif steps >= 5:
            score += 0.7
        elif steps >= 3:
            score += 0.4

        # Source citations
        citations = len(re.findall(r"\[Source:|Source:|from .{3,30} chapter|per .{3,30} §", response, re.IGNORECASE))
        if citations >= 5:
            score += 1.0
        elif citations >= 3:
            score += 0.6
        elif citations >= 1:
            score += 0.3

        # Exact compliance versions (e.g., PCI-DSS 4.0.1 §1.2)
        compliance_versions = re.findall(
            r"PCI-DSS\s+\d+\.\d+|HIPAA\s+\d{4}|SOX\s+\d{4}|NIST\s+CSF\s+\d+|ISO\s+27001:\d{4}",
            response,
            re.IGNORECASE,
        )
        if compliance_versions:
            score += 0.5

        # Config examples
        if "```" in response or "interface " in response.lower() or "router bgp" in response.lower():
            score += 0.5

        return min(5.0, score)

    def evaluate_batch(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        """Evaluate batch using local heuristics."""
        scores = []
        for result in results:
            text = result.get("reasoning", "") + "\n" + result.get("predicted", "")
            scores.append(self.score(text))

        avg_score = sum(scores) / len(scores) if scores else 0.0
        return {
            "score": avg_score,
            "num_evaluated": len(scores),
            "min_score": min(scores) if scores else 0.0,
            "max_score": max(scores) if scores else 0.0,
            "method": "local_heuristic",
        }
