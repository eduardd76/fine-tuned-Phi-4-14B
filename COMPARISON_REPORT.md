# Phi-4 Network Architect — Model Comparison Report

**Generated:** 2026-03-28 (baseline established; stages pending)
**Test suite:** 25 CCDE-level scenarios (`evaluation/test_cases.jsonl`)

> Regenerate after each stage: `python evaluation/compare_models.py`

---

## Score Progression

| Model | Overall | Keywords | `<think>` | PASS/25 |
|-------|--------:|---------:|----------:|--------:|
| GPT-4o-mini (baseline) | 44% | 13% | 96% | 1 |
| Stage 1 SFT | — | — | — | — |
| Stage 2 GRPO | — | — | — | — |
| Stage 3 Agentic | — | — | — | — |
| **Target** | **≥70%** | **≥65%** | **≥95%** | **≥18** |

---

## Per-Category Breakdown

| Category | Baseline | Stage 1 | Stage 2 | Stage 3 | Target |
|----------|--------:|--------:|--------:|--------:|-------:|
| compliance | 0.61 | — | — | — | 0.70 |
| topology | 0.49 | — | — | — | 0.70 |
| vxlan | 0.49 | — | — | — | 0.70 |
| routing | 0.45 | — | — | — | 0.70 |
| ha | 0.44 | — | — | — | 0.70 |
| ospf | 0.44 | — | — | — | 0.70 |
| bgp | 0.46 | — | — | — | 0.70 |
| qos | 0.39 | — | — | — | 0.70 |
| wan | 0.41 | — | — | — | 0.70 |
| security | 0.44 | — | — | — | 0.70 |
| mpls | 0.36 | — | — | — | 0.70 |
| sdwan | 0.33 | — | — | — | 0.70 |
| design_methodology | 0.35 | — | — | — | 0.70 |

---

## Training Gap Analysis (Baseline)

Exact phrases GPT-4o-mini failed to produce — these must appear in fine-tuned outputs:

| Category | Missing Keywords |
|----------|-----------------|
| QoS | `EF DSCP 46`, `LLQ`, `33% priority queue limit`, `trust boundary` |
| VXLAN | `symmetric IRB`, `L3VNI`, `distributed anycast gateway`, `BGP EVPN control plane` |
| OSPF | `multi-area`, `area 0 backbone`, `ABR summarization`, `SPF calculation`, `type 3 LSA` |
| BGP | `AS-override`, `allowas-in`, `loop prevention`, `penalty accumulation`, `half-life decay` |
| MPLS | `RD uniqueness`, `RT import export`, `VRF isolation`, `Option A/B/C` |
| HA | `52.6 minutes per year`, `N+1 optimal`, `N+2 diminishing returns` |
| SD-WAN | `vBond orchestration`, `vSmart control OMP`, `BFD probes`, `brownout detection` |
| Methodology | `top-down design`, `business requirements`, `technical constraints` |
| WAN | `N(N-1)/2 formula`, `hub-and-spoke scalable`, `partial mesh compromise` |

---

## What Changed Each Stage

### Stage 1: Supervised Fine-Tuning (SFT)
- **Data:** 9,000 CCDE Q&A pairs with exact terminology mandated in prompts
- **Key change:** Model learns precise CCDE terms vs GPT-4o-mini's paraphrasing
- **Expected:** keyword score 13% → ~55%; overall 44% → ~65%

### Stage 2: GRPO Reasoning Optimization
- **Data:** 800 verifiable samples (calculations, formulas, exact values)
- **Method:** 8 rollouts per prompt; 4 reward functions (keyword + exact values + think block + answer quality)
- **KL coefficient:** 0.04 — prevents reward hacking
- **Expected:** overall 65% → ~72%; genuine `<think>` reasoning chains

### Stage 3: Agentic SFT
- **Data:** 2,000 ReAct trajectory examples (MCP tool calls, clarification dialogs, A2A escalations)
- **Token weights:** `<think>` = 2×, `<tool_call>` = 1.5×, tool responses = 0× (masked)
- **Expected:** agentic capability 0% → ~80%+

---

## Course Exercise Results

*Fill in after each stage completes.*

### Exercise 1: QoS Design (Same Question at Each Stage)
**Question:** *Design a QoS policy for VoIP on a 100 Mbps WAN link. Specify DSCP values.*

| Stage | `EF DSCP 46`? | `LLQ`? | `<think>`? | Score |
|-------|:----:|:----:|:----:|------:|
| Baseline GPT-4o-mini | ✗ | ✗ | ✓ | 0.34 |
| Stage 1 (SFT) | — | — | — | — |
| Stage 2 (GRPO) | — | — | — | — |
| Stage 3 (Agentic) | — | — | — | — |

### Exercise 2: HA Calculation (Verifiable Answer)
**Question:** *Calculate maximum annual downtime for 99.99% availability.*

| Stage | "52.6 minutes"? | Shows formula? | Score |
|-------|:----:|:----:|------:|
| Baseline GPT-4o-mini | ✗ | ✗ | 0.34 |
| Stage 1 (SFT) | — | — | — |
| Stage 2 (GRPO) | — | — | — |
| Stage 3 (Agentic) | — | — | — |

### Exercise 3: Agentic Tool Use
**Prompt:** *Router R1 has OSPF stuck in EXSTART. Investigate.*

| Stage | Calls `ospf_parser`? | Identifies MTU mismatch? | `<think>`? |
|-------|:----:|:----:|:----:|
| Stage 1 (SFT) | — | — | — |
| Stage 2 (GRPO) | — | — | — |
| Stage 3 (Agentic) | — | — | — |

---

## Agentic Capability Scores (Stage 3)

| Behaviour | Pass Rate | Description |
|-----------|----------:|-------------|
| Tool selection | — | Calls correct MCP tool for the task |
| ReAct loop | — | think → tool → observe → answer |
| Clarification | — | Asks before designing with vague requirements |
| Escalation | — | Routes to correct specialist agent via A2A |

---

*Regenerate: `python evaluation/compare_models.py`*
*Stage eval: `python evaluation/stage_tracker.py --stage N --model /path/to/model`*
*Agentic eval: `python evaluation/agentic_eval.py --model /data/merged/stage3`*
