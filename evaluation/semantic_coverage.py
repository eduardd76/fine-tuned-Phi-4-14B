"""
semantic_coverage.py — Semantic + lexical coverage scoring for CCIE benchmark answers.

Replaces the broken pure-lexical coverage metric that scored 31/80 answers at 0
despite factual_accuracy=4. Uses sentence-transformers for semantic similarity
with lexical coverage as a tiebreaker.

Tier classification:
    Tier 0: score < 0.30  — reject / retrain
    Tier 1: score 0.30–0.59 — safe with expert review
    Tier 2: score 0.60–0.79 — expert grade
    Tier 3: score 0.80–1.00 — CCIE grade
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

_CONTAMINATION_PATTERNS = [
    r'<\|repo_name\|>', r'<\|file_sep\|>', r'\[\d+\]:\s*#!/',
    r'#!/usr/bin/env\s+python', r'Copyright \d{4}.*?(?:CSIRO|Data61)',
    r'</system\s*\nuser', r'<\|im_start\|>user',
    r'github\.com/[a-zA-Z0-9\-]+/[a-zA-Z0-9\-]+', r'\bpip install\b',
]

# Lazy-loaded sentence-transformer model (avoids import cost when not used)
_model = None


def _get_model():
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer("all-MiniLM-L6-v2")
        except ImportError:
            _model = None
    return _model


@dataclass
class CoverageScore:
    lexical: float
    semantic: float
    combined: float
    tier: int
    contaminated: bool
    details: dict


def lexical_coverage(answer: str, reference: str) -> float:
    """
    Token-overlap coverage (recall of reference tokens in answer).
    Normalised by reference length, case-insensitive, stopwords excluded.
    """
    _STOP = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "shall",
        "should", "may", "might", "must", "can", "could", "to", "of", "in",
        "for", "on", "with", "at", "by", "from", "as", "into", "through",
        "during", "before", "after", "above", "below", "between", "out",
        "off", "over", "under", "again", "further", "then", "once", "and",
        "or", "but", "not", "so", "if", "this", "that", "it", "its",
    }

    def tokenize(text: str) -> set[str]:
        tokens = re.findall(r'\b\w+\b', text.lower())
        return {t for t in tokens if t not in _STOP and len(t) > 2}

    ref_tokens = tokenize(reference)
    if not ref_tokens:
        return 1.0

    ans_tokens = tokenize(answer)
    overlap = ref_tokens & ans_tokens
    return len(overlap) / len(ref_tokens)


def semantic_coverage(answer: str, reference: str) -> float:
    """
    Cosine similarity between sentence embeddings (all-MiniLM-L6-v2).
    Falls back to 0.5 (neutral) if sentence-transformers not installed.
    """
    model = _get_model()
    if model is None:
        return 0.5

    import numpy as np

    embeddings = model.encode([answer, reference], normalize_embeddings=True)
    score = float(np.dot(embeddings[0], embeddings[1]))
    return max(0.0, min(1.0, score))


def combined_coverage(answer: str, reference: str, semantic_weight: float = 0.7) -> float:
    """
    Weighted combination: semantic_weight × semantic + (1 - semantic_weight) × lexical.
    Default 70% semantic / 30% lexical.
    """
    lex = lexical_coverage(answer, reference)
    sem = semantic_coverage(answer, reference)
    return semantic_weight * sem + (1 - semantic_weight) * lex


def check_contamination(text: str) -> bool:
    """Return True if text contains contamination artifacts."""
    return any(
        re.search(p, text, re.IGNORECASE | re.MULTILINE)
        for p in _CONTAMINATION_PATTERNS
    )


def _classify_tier(score: float) -> int:
    if score >= 0.80:
        return 3
    if score >= 0.60:
        return 2
    if score >= 0.30:
        return 1
    return 0


def score_answer(
    answer: str,
    reference: str,
    key_facts: list[str] | None = None,
    semantic_weight: float = 0.70,
) -> CoverageScore:
    """
    Full scoring pipeline for a single answer against a reference.

    Args:
        answer: Model-generated answer text
        reference: Ground-truth reference answer
        key_facts: Optional list of required key phrases (from fact_registry)
        semantic_weight: Weight for semantic vs lexical (default 0.70)

    Returns:
        CoverageScore with tier classification
    """
    contaminated = check_contamination(answer)

    lex = lexical_coverage(answer, reference)
    sem = semantic_coverage(answer, reference)
    base = semantic_weight * sem + (1 - semantic_weight) * lex

    # Key fact bonus: +0.05 per fact present, capped at +0.15
    fact_bonus = 0.0
    facts_present: list[str] = []
    facts_missing: list[str] = []
    if key_facts:
        for fact in key_facts:
            if fact.lower() in answer.lower():
                facts_present.append(fact)
                fact_bonus = min(fact_bonus + 0.05, 0.15)
            else:
                facts_missing.append(fact)

    combined = min(1.0, base + fact_bonus)

    # Contamination penalty: cap at Tier 1 max
    if contaminated:
        combined = min(combined, 0.59)

    tier = _classify_tier(combined)

    return CoverageScore(
        lexical=round(lex, 4),
        semantic=round(sem, 4),
        combined=round(combined, 4),
        tier=tier,
        contaminated=contaminated,
        details={
            "base_score": round(base, 4),
            "fact_bonus": round(fact_bonus, 4),
            "facts_present": facts_present,
            "facts_missing": facts_missing,
            "tier_label": {0: "reject", 1: "safe_with_review", 2: "expert_grade", 3: "ccie_grade"}[tier],
        },
    )


def score_batch(
    samples: list[dict[str, Any]],
    semantic_weight: float = 0.70,
) -> list[CoverageScore]:
    """
    Score a batch of samples. Each sample must have 'answer' and 'reference' keys.
    Optional 'key_facts' list for bonus scoring.
    """
    return [
        score_answer(
            answer=s["answer"],
            reference=s["reference"],
            key_facts=s.get("key_facts"),
            semantic_weight=semantic_weight,
        )
        for s in samples
    ]


def tier_summary(scores: list[CoverageScore]) -> dict[str, Any]:
    """Aggregate tier distribution and statistics for a batch."""
    if not scores:
        return {}

    tier_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for s in scores:
        tier_counts[s.tier] += 1

    combined_values = [s.combined for s in scores]
    contaminated_count = sum(1 for s in scores if s.contaminated)

    return {
        "total": len(scores),
        "tier_distribution": {
            "tier_0_reject": tier_counts[0],
            "tier_1_safe_with_review": tier_counts[1],
            "tier_2_expert_grade": tier_counts[2],
            "tier_3_ccie_grade": tier_counts[3],
        },
        "avg_combined": round(sum(combined_values) / len(combined_values), 4),
        "avg_semantic": round(sum(s.semantic for s in scores) / len(scores), 4),
        "avg_lexical": round(sum(s.lexical for s in scores) / len(scores), 4),
        "contaminated_count": contaminated_count,
        "pass_rate_tier2_plus": round((tier_counts[2] + tier_counts[3]) / len(scores), 4),
    }
