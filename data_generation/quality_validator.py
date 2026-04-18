"""
quality_validator.py — 6-check validation for LLM-generated training samples.
Rejects samples that don't meet quality thresholds. Called after each LLM generation.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Any

CONTAMINATION_PATTERNS = [
    r'<\|repo_name\|>', r'<\|file_sep\|>', r'\[\d+\]:\s*#!/',
    r'#!/usr/bin/env\s+python', r'Copyright \d{4}.*?(?:CSIRO|Data61)',
    r'</system\s*\nuser', r'<\|im_start\|>user',
    r'github\.com/[a-zA-Z0-9\-]+/[a-zA-Z0-9\-]+', r'\bpip install\b',
    r'^\s*from [a-z_]+ import |^\s*import [a-z_]+$', r'\bMakefile\b.*\bCFLAGS\b',
]


def has_no_contamination(text: str) -> bool:
    """Return True if text contains no training data contamination artifacts."""
    return not any(
        re.search(p, text, re.IGNORECASE | re.MULTILINE)
        for p in CONTAMINATION_PATTERNS
    )


@dataclass
class ValidationResult:
    passed: bool
    checks: dict = field(default_factory=dict)
    failures: list = field(default_factory=list)
    score: float = 0.0


def validate_sample(sample: dict, sub_type: str = "") -> ValidationResult:
    """
    Run 6 quality checks on a training sample.
    Returns ValidationResult with passed=True only if ALL checks pass.
    """
    # Import here to avoid circular imports
    from term_registry import get_required_terms

    result = ValidationResult(passed=False)

    messages = sample.get("messages", [])
    assistant_turns = [m["content"] for m in messages if m.get("role") == "assistant"]
    full_response = " ".join(assistant_turns)

    # ── Check 1: Has <think> block ────────────────────────────────────────────
    has_think = bool(re.search(r"<think>.*?</think>", full_response, re.DOTALL))
    result.checks["has_think_block"] = has_think
    if not has_think:
        result.failures.append("Missing <think>...</think> reasoning block")

    # ── Check 2: Think block has substance ───────────────────────────────────
    think_match = re.search(r"<think>(.*?)</think>", full_response, re.DOTALL)
    think_text = think_match.group(1).strip() if think_match else ""
    think_substantial = len(think_text.split()) >= 30
    result.checks["think_substantial"] = think_substantial
    if not think_substantial:
        result.failures.append(f"<think> block too short: {len(think_text.split())} words (min 30)")

    # ── Check 3: Minimum response length ─────────────────────────────────────
    answer_text = re.sub(r"<think>.*?</think>", "", full_response, flags=re.DOTALL)
    word_count = len(answer_text.split())
    min_length_ok = word_count >= 80
    result.checks["min_response_length"] = min_length_ok
    if not min_length_ok:
        result.failures.append(f"Response too short: {word_count} words (min 80)")

    # ── Check 4: No hallucination markers ────────────────────────────────────
    hallucination_patterns = [
        r"I'm not sure",
        r"I don't know",
        r"I cannot answer",
        r"As an AI",
        r"I don't have access",
        r"my knowledge cutoff",
    ]
    no_hallucination = not any(
        re.search(p, full_response, re.IGNORECASE) for p in hallucination_patterns
    )
    result.checks["no_hallucination_markers"] = no_hallucination
    if not no_hallucination:
        result.failures.append("Contains uncertainty/hallucination marker phrases")

    # ── Check 5: Required CCDE terms present ─────────────────────────────────
    required_terms = get_required_terms(sub_type) if sub_type else []
    if required_terms:
        present = [t for t in required_terms if t.lower() in full_response.lower()]
        term_coverage = len(present) / len(required_terms)
        terms_ok = term_coverage >= 0.70
        result.checks["required_terms_coverage"] = terms_ok
        missing = [t for t in required_terms if t.lower() not in full_response.lower()]
        if not terms_ok:
            result.failures.append(
                f"Term coverage {term_coverage:.0%} < 70%. Missing: {missing[:3]}"
            )
    else:
        result.checks["required_terms_coverage"] = True

    # ── Check 6: Has concrete recommendation ─────────────────────────────────
    vague_patterns = [r"\bit depends\b", r"\bvaries\b", r"\bsituation-dependent\b"]
    resolution_patterns = [
        r"recommend", r"should use", r"prefer", r"choose",
        r"select", r"design", r"implement",
    ]
    has_vague = any(re.search(p, full_response, re.IGNORECASE) for p in vague_patterns)
    has_resolution = any(re.search(p, full_response, re.IGNORECASE) for p in resolution_patterns)
    vague_ok = (not has_vague) or has_resolution
    result.checks["has_recommendation"] = vague_ok
    if not vague_ok:
        result.failures.append("Uses 'it depends' without concrete recommendation")

    # ── Check 7: No training data contamination ───────────────────────────────
    clean = has_no_contamination(full_response)
    result.checks["no_contamination"] = clean
    if not clean:
        result.failures.append("Response contains training data contamination artifacts")

    passed_count = sum(result.checks.values())
    result.score = passed_count / len(result.checks)
    result.passed = result.score == 1.0

    return result


def validate_agentic_sample(sample: dict) -> ValidationResult:
    """Validate an agentic trajectory sample (ReAct format)."""
    result = ValidationResult(passed=False)
    messages = sample.get("messages", [])
    metadata = sample.get("metadata", {})

    # ── Check 1: Has tool call ────────────────────────────────────────────────
    has_tool_call = any(
        "<tool_call>" in m.get("content", "")
        for m in messages if m.get("role") == "assistant"
    )
    result.checks["has_tool_call"] = has_tool_call
    if not has_tool_call:
        result.failures.append("Agentic sample missing <tool_call> block")

    # ── Check 2: Has tool response ────────────────────────────────────────────
    has_tool_response = any(m.get("role") == "tool" for m in messages)
    result.checks["has_tool_response"] = has_tool_response
    if not has_tool_response:
        result.failures.append("Agentic sample missing tool response message")

    # ── Check 3: Has <think> in at least one assistant turn ──────────────────
    has_think = any(
        "<think>" in m.get("content", "")
        for m in messages if m.get("role") == "assistant"
    )
    result.checks["has_think_block"] = has_think
    if not has_think:
        result.failures.append("Agentic sample missing <think> block")

    # ── Check 4: Final assistant turn has substantive answer ─────────────────
    final_assistant = next(
        (m["content"] for m in reversed(messages) if m.get("role") == "assistant"),
        ""
    )
    final_answer_ok = len(final_assistant.split()) >= 50
    result.checks["final_answer_substantial"] = final_answer_ok
    if not final_answer_ok:
        result.failures.append(f"Final assistant turn too short: {len(final_assistant.split())} words")

    # ── Check 5: Tools used are valid MCP tools ───────────────────────────────
    valid_tools = {
        "ospf_parser", "interface_parser", "connectivity_tester",
        "network_design", "security_parser", "bgp_parser",
    }
    tools_used = set(metadata.get("tools_used", []))
    tools_valid = tools_used.issubset(valid_tools) if tools_used else True
    result.checks["valid_tools_used"] = tools_valid
    if not tools_valid:
        invalid = tools_used - valid_tools
        result.failures.append(f"Invalid MCP tools referenced: {invalid}")

    # ── Check 6: Multi-turn structure ─────────────────────────────────────────
    roles = [m.get("role") for m in messages]
    has_multi_turn = "tool" in roles and roles.count("assistant") >= 2
    result.checks["multi_turn_structure"] = has_multi_turn
    if not has_multi_turn:
        result.failures.append("Agentic sample needs ≥2 assistant turns and 1 tool turn")

    # ── Check 7: No training data contamination ───────────────────────────────
    all_content = " ".join(m.get("content", "") for m in messages)
    clean = has_no_contamination(all_content)
    result.checks["no_contamination"] = clean
    if not clean:
        result.failures.append("Sample contains training data contamination artifacts")

    passed_count = sum(result.checks.values())
    result.score = passed_count / len(result.checks)
    result.passed = result.score == 1.0
    return result
