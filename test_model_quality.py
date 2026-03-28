"""
Phi-4 Network Architect — Model Quality Tester

Three modes:
  python test_model_quality.py --interactive          Ask ad-hoc questions
  python test_model_quality.py --suite               Run all 25 CCDE test cases, print scored report
  python test_model_quality.py --suite --save        Same, save results to quality_report.json

Backends (auto-detected, or force with --backend):
  api       POST http://localhost:8000/api/v1/design   (fine-tuned model via FastAPI)
  ollama    http://localhost:11434                      (local Ollama, any model)
  openai    OpenAI API                                  (comparison baseline)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

# ── optional colour ────────────────────────────────────────────────────────
try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init()
    GREEN  = Fore.GREEN
    YELLOW = Fore.YELLOW
    RED    = Fore.RED
    CYAN   = Fore.CYAN
    BOLD   = Style.BRIGHT
    RESET  = Style.RESET_ALL
except ImportError:
    GREEN = YELLOW = RED = CYAN = BOLD = RESET = ""


# ═══════════════════════════════════════════════════════════════════════════
# Data model
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ModelResponse:
    raw: str
    reasoning: str          # content inside <think>...</think>
    answer: str             # text after </think> (or full text if no block)
    has_think: bool
    reasoning_steps: int
    sources: list[str]
    latency_ms: float
    backend: str
    confidence: float = 0.0

    def __post_init__(self):
        self.confidence = _compute_confidence(self)


@dataclass
class TestResult:
    test_id: str
    category: str
    question: str
    response: ModelResponse
    keywords_found: list[str]
    keywords_missing: list[str]
    keyword_score: float        # 0.0 – 1.0
    sources_found: list[str]
    source_score: float         # 0.0 – 1.0
    overall_score: float        # weighted average
    pass_fail: str              # PASS / WARN / FAIL


# ═══════════════════════════════════════════════════════════════════════════
# Parsing helpers
# ═══════════════════════════════════════════════════════════════════════════

_THINK_RE   = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_STEP_RE    = re.compile(r"\bStep\s+\d+", re.IGNORECASE)
_BRACKET_RE = re.compile(r"\[([^\]]{4,80})\]")
_RFC_RE     = re.compile(r"\bRFC\s+\d{3,5}\b", re.IGNORECASE)
_STD_RE     = re.compile(r"\b(PCI-DSS|HIPAA|NIST CSF|IEEE 802\.\S+|3GPP|ITU-T)\b", re.IGNORECASE)


def _parse_response(raw: str, latency_ms: float, backend: str) -> ModelResponse:
    think_match = _THINK_RE.search(raw)
    if think_match:
        reasoning = think_match.group(1).strip()
        answer    = _THINK_RE.sub("", raw).strip()
        has_think = True
    else:
        reasoning = ""
        answer    = raw.strip()
        has_think = False

    steps   = len(_STEP_RE.findall(reasoning or answer))
    sources = (
        _BRACKET_RE.findall(raw)
        + _RFC_RE.findall(raw)
        + _STD_RE.findall(raw)
    )

    return ModelResponse(
        raw=raw,
        reasoning=reasoning,
        answer=answer,
        has_think=has_think,
        reasoning_steps=steps,
        sources=list(dict.fromkeys(sources)),  # deduplicate, preserve order
        latency_ms=latency_ms,
        backend=backend,
    )


def _compute_confidence(r: ModelResponse) -> float:
    score = 0.35  # base
    if r.has_think:
        score += 0.20
    if r.reasoning_steps >= 3:
        score += 0.15
    elif r.reasoning_steps >= 1:
        score += 0.08
    if r.sources:
        score += 0.15
    answer_len = len(r.answer.split())
    if answer_len >= 200:
        score += 0.10
    elif answer_len >= 80:
        score += 0.05
    return min(round(score, 2), 1.0)


# ═══════════════════════════════════════════════════════════════════════════
# Backends
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a CCDE-level network architect with deep expertise in enterprise \
and service provider network design, BGP, OSPF, MPLS, SD-WAN, VXLAN BGP EVPN, and compliance \
frameworks (PCI-DSS 4.0.1, HIPAA, NIST CSF 2.0).

Think through problems step by step inside <think> tags before your final answer.
Cite specific RFCs, standards, and Cisco documentation where relevant.
Include IOS configuration examples when helpful."""


def _query_api(question: str, api_url: str, api_key: str) -> ModelResponse:
    """POST to the FastAPI /design endpoint."""
    import urllib.request, urllib.error

    payload = json.dumps({
        "users": 0,
        "sites": 1,
        "uptime": 99.9,
        "custom_question": question,
        # The /design endpoint accepts extra fields
    }).encode()

    # Try /troubleshoot first (more general), fall back to /design
    for path in ["/api/v1/troubleshoot", "/api/v1/design"]:
        req = urllib.request.Request(
            api_url.rstrip("/") + path,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "X-API-Key": api_key,
            },
            method="POST",
        )
        try:
            t0 = time.perf_counter()
            with urllib.request.urlopen(req, timeout=120) as resp:
                latency = (time.perf_counter() - t0) * 1000
                data = json.loads(resp.read())
            # Extract text from response
            raw = data.get("diagnosis") or data.get("design") or str(data)
            reasoning = data.get("reasoning", "")
            if reasoning:
                raw = f"<think>{reasoning}</think>\n{raw}"
            return _parse_response(raw, latency, "api")
        except urllib.error.HTTPError:
            continue
        except Exception as exc:
            raise RuntimeError(f"API error: {exc}") from exc

    raise RuntimeError("API returned errors on all endpoints")


def _query_ollama(question: str, model: str, base_url: str) -> ModelResponse:
    import urllib.request

    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": question},
        ],
        "stream": False,
        "options": {"temperature": 0.7, "num_predict": 2048},
    }).encode()

    req = urllib.request.Request(
        base_url.rstrip("/") + "/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as resp:
        latency = (time.perf_counter() - t0) * 1000
        data = json.loads(resp.read())
    raw = data["message"]["content"]
    return _parse_response(raw, latency, f"ollama/{model}")


def _query_openai(question: str, model: str, api_key: str) -> ModelResponse:
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": question},
        ],
        max_tokens=2048,
        temperature=0.7,
    )
    latency = (time.perf_counter() - t0) * 1000
    raw = resp.choices[0].message.content or ""
    return _parse_response(raw, latency, f"openai/{model}")


# ═══════════════════════════════════════════════════════════════════════════
# Backend auto-detection
# ═══════════════════════════════════════════════════════════════════════════

def _detect_backend(force: str | None) -> dict:
    """Return a dict describing the backend to use."""
    import urllib.request, urllib.error

    if force == "api" or force is None:
        try:
            with urllib.request.urlopen("http://localhost:8000/api/v1/health", timeout=3):
                return {"type": "api", "url": "http://localhost:8000",
                        "key": os.environ.get("PHI4_API_KEY", "dev-key")}
        except Exception:
            if force == "api":
                print(f"{RED}API not reachable at localhost:8000{RESET}")
                sys.exit(1)

    if force == "ollama" or force is None:
        ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
        try:
            with urllib.request.urlopen(f"{ollama_url}/api/version", timeout=3) as r:
                # Pick best available model
                with urllib.request.urlopen(f"{ollama_url}/api/tags", timeout=3) as r2:
                    tags = json.loads(r2.read())
                models = [m["name"] for m in tags.get("models", [])]
                preferred = ["phi4", "phi3", "llama3", "mistral", "gemma"]
                model = next((m for p in preferred for m in models if p in m.lower()), None)
                model = model or (models[0] if models else "mistral")
                return {"type": "ollama", "url": ollama_url, "model": model}
        except Exception:
            if force == "ollama":
                print(f"{RED}Ollama not reachable at {ollama_url}{RESET}")
                sys.exit(1)

    if force == "openai" or force is None:
        key = os.environ.get("OPENAI_API_KEY")
        if key:
            return {"type": "openai", "model": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"), "key": key}
        if force == "openai":
            print(f"{RED}OPENAI_API_KEY not set{RESET}")
            sys.exit(1)

    print(f"{RED}No backend available. Start the API, Ollama, or set OPENAI_API_KEY.{RESET}")
    sys.exit(1)


def query(question: str, backend: dict) -> ModelResponse:
    t = backend["type"]
    if t == "api":
        return _query_api(question, backend["url"], backend["key"])
    if t == "ollama":
        return _query_ollama(question, backend["model"], backend["url"])
    if t == "openai":
        return _query_openai(question, backend["model"], backend["key"])
    raise ValueError(f"Unknown backend type: {t}")


# ═══════════════════════════════════════════════════════════════════════════
# Display helpers
# ═══════════════════════════════════════════════════════════════════════════

def _bar(score: float, width: int = 20) -> str:
    filled = round(score * width)
    bar    = "█" * filled + "░" * (width - filled)
    colour = GREEN if score >= 0.75 else (YELLOW if score >= 0.50 else RED)
    return f"{colour}{bar}{RESET} {score:.0%}"


def _pf_colour(pf: str) -> str:
    if pf == "PASS":
        return f"{GREEN}{BOLD}{pf}{RESET}"
    if pf == "WARN":
        return f"{YELLOW}{BOLD}{pf}{RESET}"
    return f"{RED}{BOLD}{pf}{RESET}"


def print_response(r: ModelResponse, verbose: bool = True) -> None:
    print(f"\n{CYAN}{'─'*70}{RESET}")
    print(f"{BOLD}Backend:{RESET}  {r.backend}  |  "
          f"{BOLD}Latency:{RESET} {r.latency_ms/1000:.1f}s  |  "
          f"{BOLD}Confidence:{RESET} {_bar(r.confidence, 12)}")

    if r.has_think and verbose:
        print(f"\n{YELLOW}{BOLD}[ Reasoning ]{RESET}")
        # Truncate very long reasoning in display
        lines = r.reasoning.split("\n")
        display = "\n".join(lines[:30])
        if len(lines) > 30:
            display += f"\n  ... ({len(lines)-30} more lines)"
        print(display)

    elif not r.has_think:
        print(f"  {YELLOW}⚠  No <think> block — model may not be fine-tuned{RESET}")

    print(f"\n{BOLD}[ Answer ]{RESET}")
    print(r.answer[:3000] + ("..." if len(r.answer) > 3000 else ""))

    if r.sources:
        print(f"\n{BOLD}[ Sources cited ]{RESET}")
        for s in r.sources[:8]:
            print(f"  • {s}")

    print(f"\n{BOLD}Reasoning steps:{RESET} {r.reasoning_steps}  |  "
          f"{BOLD}<think> block:{RESET} {'✓' if r.has_think else '✗'}")


# ═══════════════════════════════════════════════════════════════════════════
# Scoring
# ═══════════════════════════════════════════════════════════════════════════

def _score_response(tc: dict, r: ModelResponse) -> TestResult:
    text_lower = r.raw.lower()

    expected_kw  = tc.get("expected_key_facts", [])
    expected_src = tc.get("sources_required", [])

    kw_found   = [k for k in expected_kw  if k.lower() in text_lower]
    kw_missing = [k for k in expected_kw  if k.lower() not in text_lower]
    src_found  = [s for s in expected_src if any(w.lower() in text_lower
                                                  for w in s.split() if len(w) > 4)]

    kw_score  = len(kw_found)  / max(len(expected_kw),  1)
    src_score = len(src_found) / max(len(expected_src), 1)

    think_bonus = 0.10 if r.has_think else 0.0
    overall = round(0.50 * kw_score + 0.20 * src_score
                    + 0.20 * r.confidence + think_bonus, 3)

    if overall >= 0.70:
        pf = "PASS"
    elif overall >= 0.45:
        pf = "WARN"
    else:
        pf = "FAIL"

    return TestResult(
        test_id       = tc["id"],
        category      = tc.get("category", "?"),
        question      = tc["messages"][0]["content"],
        response      = r,
        keywords_found   = kw_found,
        keywords_missing = kw_missing,
        keyword_score    = kw_score,
        sources_found    = src_found,
        source_score     = src_score,
        overall_score    = overall,
        pass_fail        = pf,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Modes
# ═══════════════════════════════════════════════════════════════════════════

def run_interactive(backend: dict) -> None:
    """REPL — ask free-form questions."""
    b_label = backend.get("model") or backend.get("url") or backend["type"]
    print(f"\n{BOLD}Phi-4 Network Architect — Interactive Mode{RESET}")
    print(f"Backend: {CYAN}{backend['type']} ({b_label}){RESET}")
    print("Type your question and press Enter. Empty line to quit.\n")

    while True:
        try:
            q = input(f"{BOLD}Q> {RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not q:
            break
        try:
            r = query(q, backend)
            print_response(r, verbose=True)
        except Exception as exc:
            print(f"{RED}Error: {exc}{RESET}")


def run_suite(backend: dict, save: bool, filter_cat: str | None) -> None:
    """Run all 25 test cases, print a scored report."""
    tc_path = Path(__file__).parent / "evaluation" / "test_cases.jsonl"
    if not tc_path.exists():
        print(f"{RED}Test cases not found: {tc_path}{RESET}")
        sys.exit(1)

    test_cases = []
    with tc_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                test_cases.append(json.loads(line))

    if filter_cat:
        test_cases = [t for t in test_cases if t.get("category", "").lower() == filter_cat.lower()]
        if not test_cases:
            print(f"{RED}No test cases found for category '{filter_cat}'{RESET}")
            sys.exit(1)

    b_label = backend.get("model") or backend.get("url") or backend["type"]
    print(f"\n{BOLD}Running {len(test_cases)} test cases against {backend['type']} ({b_label}){RESET}\n")

    results: list[TestResult] = []
    for i, tc in enumerate(test_cases, 1):
        q = tc["messages"][0]["content"]
        short_q = q[:80] + "…" if len(q) > 80 else q
        print(f"  [{i:>2}/{len(test_cases)}] {tc['id']} ({tc.get('category','?')}) — {short_q}", end="", flush=True)

        try:
            r = query(q, backend)
            tr = _score_response(tc, r)
            results.append(tr)
            pf_str = _pf_colour(tr.pass_fail)
            print(f"\r  [{i:>2}/{len(test_cases)}] {tc['id']} "
                  f"score={tr.overall_score:.2f} {pf_str} "
                  f"kw={tr.keyword_score:.0%} "
                  f"think={'✓' if r.has_think else '✗'} "
                  f"{r.latency_ms/1000:.1f}s")
        except Exception as exc:
            print(f"\n    {RED}ERROR: {exc}{RESET}")

    _print_report(results)

    if save:
        out = Path("quality_report.json")
        out.write_text(json.dumps([
            {
                "test_id":       tr.test_id,
                "category":      tr.category,
                "overall_score": tr.overall_score,
                "keyword_score": tr.keyword_score,
                "source_score":  tr.source_score,
                "confidence":    tr.response.confidence,
                "has_think":     tr.response.has_think,
                "reasoning_steps": tr.response.reasoning_steps,
                "latency_ms":    tr.response.latency_ms,
                "pass_fail":     tr.pass_fail,
                "keywords_missing": tr.keywords_missing,
            }
            for tr in results
        ], indent=2))
        print(f"\n{GREEN}Saved → {out.resolve()}{RESET}")


def _print_report(results: list[TestResult]) -> None:
    if not results:
        return

    passes = sum(1 for r in results if r.pass_fail == "PASS")
    warns  = sum(1 for r in results if r.pass_fail == "WARN")
    fails  = sum(1 for r in results if r.pass_fail == "FAIL")
    avg_overall = sum(r.overall_score    for r in results) / len(results)
    avg_kw      = sum(r.keyword_score    for r in results) / len(results)
    avg_conf    = sum(r.response.confidence for r in results) / len(results)
    avg_lat     = sum(r.response.latency_ms for r in results) / len(results)
    think_rate  = sum(1 for r in results if r.response.has_think) / len(results)

    print(f"\n{'═'*70}")
    print(f"{BOLD}QUALITY REPORT — {len(results)} test cases{RESET}")
    print(f"{'═'*70}")
    print(f"  Overall score   {_bar(avg_overall)}")
    print(f"  Keyword match   {_bar(avg_kw)}")
    print(f"  Confidence      {_bar(avg_conf)}")
    print(f"  <think> rate    {_bar(think_rate)}")
    print(f"  Avg latency     {avg_lat/1000:.1f}s")
    print(f"\n  {GREEN}PASS{RESET}: {passes}   {YELLOW}WARN{RESET}: {warns}   {RED}FAIL{RESET}: {fails}")

    # Per-category breakdown
    cats: dict[str, list[TestResult]] = {}
    for r in results:
        cats.setdefault(r.category, []).append(r)

    print(f"\n{'─'*70}")
    print(f"{'Category':<22} {'Cases':>5} {'Score':>7} {'Keywords':>9} {'Think':>7}")
    print(f"{'─'*70}")
    for cat, rs in sorted(cats.items()):
        s  = sum(r.overall_score for r in rs) / len(rs)
        kw = sum(r.keyword_score for r in rs) / len(rs)
        th = sum(1 for r in rs if r.response.has_think) / len(rs)
        col = GREEN if s >= 0.70 else (YELLOW if s >= 0.45 else RED)
        print(f"  {cat:<20} {len(rs):>5}  {col}{s:>6.2f}{RESET}   {kw:>7.0%}   {th:>6.0%}")

    # Failures
    failed = [r for r in results if r.pass_fail == "FAIL"]
    if failed:
        print(f"\n{RED}{BOLD}Failed tests:{RESET}")
        for r in failed:
            print(f"  {r.test_id} ({r.category}) score={r.overall_score:.2f}")
            if r.keywords_missing:
                print(f"    Missing keywords: {', '.join(r.keywords_missing)}")

    # Top missing keywords (common gaps)
    all_missing: dict[str, int] = {}
    for r in results:
        for kw in r.keywords_missing:
            all_missing[kw] = all_missing.get(kw, 0) + 1
    if all_missing:
        top = sorted(all_missing.items(), key=lambda x: -x[1])[:8]
        print(f"\n{YELLOW}Most missed keywords (training gaps):{RESET}")
        for kw, cnt in top:
            print(f"  {cnt}x  {kw}")


def run_single(test_id: str, backend: dict) -> None:
    """Run one specific test case by ID, show full response."""
    tc_path = Path(__file__).parent / "evaluation" / "test_cases.jsonl"
    with tc_path.open() as f:
        cases = [json.loads(l) for l in f if l.strip()]

    tc = next((c for c in cases if c["id"] == test_id), None)
    if not tc:
        ids = [c["id"] for c in cases]
        print(f"{RED}Test ID '{test_id}' not found. Available: {', '.join(ids)}{RESET}")
        sys.exit(1)

    q = tc["messages"][0]["content"]
    print(f"\n{BOLD}Question:{RESET} {q}\n")
    r  = query(q, backend)
    tr = _score_response(tc, r)

    print_response(r, verbose=True)
    print(f"\n{'─'*50}")
    print(f"{BOLD}Score:{RESET}   {_bar(tr.overall_score)}")
    print(f"Keywords: {tr.keyword_score:.0%}  ({len(tr.keywords_found)}/{len(tr.keywords_found)+len(tr.keywords_missing)})")
    if tr.keywords_found:
        print(f"  {GREEN}Found:{RESET}   {', '.join(tr.keywords_found)}")
    if tr.keywords_missing:
        print(f"  {RED}Missing:{RESET} {', '.join(tr.keywords_missing)}")
    print(f"Result:  {_pf_colour(tr.pass_fail)}")


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    p = argparse.ArgumentParser(
        description="Test Phi-4 Network Architect model quality",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_model_quality.py --interactive
  python test_model_quality.py --suite
  python test_model_quality.py --suite --category bgp
  python test_model_quality.py --suite --save --backend openai
  python test_model_quality.py --test-id tc_004
  python test_model_quality.py --ask "Design a spine-leaf fabric for 500 servers"
""",
    )
    p.add_argument("--interactive",  action="store_true",  help="Ask questions in a REPL")
    p.add_argument("--suite",        action="store_true",  help="Run all 25 CCDE test cases")
    p.add_argument("--save",         action="store_true",  help="Save suite results to quality_report.json")
    p.add_argument("--category",     metavar="CAT",        help="Filter suite by category (bgp/ospf/vxlan/...)")
    p.add_argument("--test-id",      metavar="ID",         help="Run a single test case, e.g. tc_004")
    p.add_argument("--ask",          metavar="QUESTION",   help="Ask a single question and exit")
    p.add_argument("--backend",      choices=["api","ollama","openai"],
                                                           help="Force a specific backend")
    p.add_argument("--ollama-model", metavar="MODEL",      help="Ollama model name (default: auto-detect)")
    p.add_argument("--openai-model", default="gpt-4o-mini",help="OpenAI model (default: gpt-4o-mini)")
    p.add_argument("--verbose", "-v", action="store_true", help="Show full reasoning block in suite mode")

    args = p.parse_args()

    if not any([args.interactive, args.suite, args.test_id, args.ask]):
        p.print_help()
        sys.exit(0)

    backend = _detect_backend(args.backend)

    # Apply overrides
    if args.ollama_model and backend["type"] == "ollama":
        backend["model"] = args.ollama_model
    if backend["type"] == "openai":
        backend["model"] = args.openai_model

    b_label = backend.get("model") or backend["type"]
    print(f"{BOLD}Backend:{RESET} {CYAN}{backend['type']}{RESET} ({b_label})")

    if args.ask:
        r = query(args.ask, backend)
        print_response(r, verbose=True)

    elif args.interactive:
        run_interactive(backend)

    elif args.suite:
        run_suite(backend, args.save, args.category)

    elif args.test_id:
        run_single(args.test_id, backend)


if __name__ == "__main__":
    main()
