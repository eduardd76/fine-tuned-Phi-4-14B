# Fine-Tuning Phi-4-14B as an Agentic Network Architect
## Complete Course Documentation

**Project:** Dream Team Virtual Architect Agent
**Model:** Microsoft Phi-4-14B → `phi4-network-architect`
**Goal:** A CCDE-level network architect that reasons, uses tools, and operates autonomously inside a multi-agent system

---

## Why This Is Different From Standard Fine-Tuning

Most fine-tuning courses teach you to make a model answer questions better.

This is different. The end product is not a chatbot — it is an **autonomous agent** that:
- Receives tasks from a Team Leader via Redis (A2A protocol)
- Decides what information it needs before answering
- Calls MCP tools to gather live network state
- Reasons through multi-step design problems
- Knows when to ask for clarification vs. when to proceed
- Knows when to escalate to a human vs. when to act autonomously
- Produces structured responses consumed by other agents and the UI

**This changes everything about how the training data must be built.**

A standard fine-tuned model learns: `Question → Answer`

An agentic fine-tuned model learns: `Task → Think → Decide → (Tool Call → Observe →)* → Think → Respond → (Escalate?)`

---

## Table of Contents

1. [The Problem We Are Solving](#1-the-problem-we-are-solving)
2. [Why Phi-4-14B](#2-why-phi-4-14b)
3. [Architecture Position of This Model](#3-architecture-position-of-this-model)
4. [The Three-Stage Training Plan](#4-the-three-stage-training-plan)
5. [Stage 1 — Dataset Design](#5-stage-1--dataset-design)
6. [Stage 1 — SFT Training Strategy](#6-stage-1--sft-training-strategy)
7. [Stage 2 — GRPO for Reasoning](#7-stage-2--grpo-for-reasoning)
8. [Stage 3 — Agentic Behaviour Training](#8-stage-3--agentic-behaviour-training)
9. [Evaluation Framework](#9-evaluation-framework)
10. [Deployment Strategy](#10-deployment-strategy)
11. [Key Decisions Log](#11-key-decisions-log)
12. [Baseline Results](#12-baseline-results)
13. [Course Exercises](#13-course-exercises)

---

## 1. The Problem We Are Solving

### Business context

The Dream Team agentic-netops-mvp automates network operations for Cisco enterprise environments. When a security incident or network failure occurs, AI agents diagnose the problem and propose fixes — with human approval before any command is executed.

One agent — the **Virtual Architect** — handles all design and architecture questions:
- "This network is failing. Should we redesign the WAN topology?"
- "We need to add 2,000 users at a new site. Design the extension."
- "Our MPLS VPN Inter-AS is having issues. What's the fix?"

### Why a general LLM is not enough

GPT-4o can answer these questions, but:

| Requirement | GPT-4o | Fine-tuned Phi-4 |
|-------------|--------|-----------------|
| Runs on-prem (no data leaves) | ❌ | ✅ |
| Exact CCDE terminology for eval scoring | ❌ 13% | ✅ target 70%+ |
| Native `<think>` chain-of-thought | ❌ (system prompt hack) | ✅ (trained behaviour) |
| Tool-calling in A2A task format | ❌ | ✅ |
| Knows when to escalate vs. act | ❌ | ✅ |
| Cost at scale (30 req/min sustained) | ❌ expensive | ✅ free after training |
| Response speed (3–6s vs 15s) | ❌ | ✅ |

### What the model must be able to do

```
Agentic capability checklist:

[ ] Receive an A2A task (JSON) and extract the design request
[ ] Identify missing requirements and ask for them before proceeding
[ ] Decompose a complex design into sub-problems
[ ] Decide which MCP tools to call to gather current network state
[ ] Call tools, observe results, update reasoning
[ ] Produce a design recommendation with explicit CCDE reasoning
[ ] Cite specific RFCs, standards, and Cisco CVDs
[ ] Assess its own confidence and flag low-confidence responses
[ ] Recognise compliance requirements (PCI-DSS, HIPAA) and escalate
[ ] Format the response as a structured A2A reply
[ ] Maintain context across a multi-turn design session
```

---

## 2. Why Phi-4-14B

### Model selection criteria

When choosing the base model, we evaluated five candidates:

| Model | Size | `<think>` support | Tool use | VRAM (4-bit) | Decision |
|-------|------|-------------------|----------|--------------|----------|
| Llama-3.1-8B | 8B | ❌ needs training | Via prompt | ~5 GB | Too small for CCDE depth |
| Mistral-7B-v0.3 | 7B | ❌ | Via prompt | ~4 GB | Too small |
| Qwen2.5-14B | 14B | Partial | ✅ native | ~9 GB | Strong candidate |
| **Phi-4-14B** | **14B** | **✅ native** | **✅ native** | **~9 GB** | **Selected** |
| Llama-3.1-70B | 70B | ❌ | Via prompt | ~40 GB | Too large for target hardware |

### Why Phi-4 specifically

**1. Native reasoning tokens**

Phi-4 was trained with `<think>...</think>` reasoning blocks as a first-class feature — the same mechanism used by DeepSeek-R1 and Qwen-QwQ. This is critical because:
- The tokenizer already has dedicated think tokens (no hallucinated format)
- We can apply 2× loss weight on think tokens during SFT (reinforces reasoning)
- GRPO naturally rewards longer, structured reasoning chains

**2. Strong technical knowledge baseline**

Phi-4 was trained on a high proportion of technical documentation, textbooks, and code. Network engineering is well-represented. This means:
- Domain adaptation (not from scratch) — much less data needed
- The model already knows what BGP is; we're teaching it CCDE-depth analysis

**3. 14B is the right size for the target hardware**

14B at 4-bit quantization fits in ~9 GB VRAM — within a T4 (16 GB) or A10G (24 GB) EC2 instance. Inference at 3–6 seconds per request is acceptable for the A2A workflow.

**4. Native tool-calling format**

Phi-4's instruction tuning includes structured tool-call syntax, which maps directly to the MCP tool calls the agent needs to make.

### What we are NOT doing

- Not using Phi-4-mini (3.8B) — insufficient depth for CCDE reasoning
- Not using Phi-4-multimodal — unnecessary complexity
- Not using a larger model — deployment cost/complexity not justified

---

## 3. Architecture Position of This Model

Understanding where this model sits in the system is essential for understanding what the training data must teach it.

```
┌─────────────────────────────────────────────────────┐
│  Layer 1: Human (Streamlit UI)                      │
│  "Should we redesign the WAN for the new merger?"   │
└─────────────────┬───────────────────────────────────┘
                  │ ACP (HTTP)
┌─────────────────▼───────────────────────────────────┐
│  Layer 2: Team Leader Agent                         │
│  Triages: "This is a design question → Virtual      │
│  Architect"                                         │
│  Sends A2A task → Redis: a2a_virtual_architect_inbox│
└─────────────────┬───────────────────────────────────┘
                  │ A2A (Redis queue)
┌─────────────────▼───────────────────────────────────┐
│  Layer 3: Virtual Architect (THIS MODEL)            │
│                                                     │
│  1. Reads A2A task from Redis                       │
│  2. Parses requirements                             │
│  3. <think> through the design problem              │
│  4. Calls MCP tools if current network state needed │
│  5. Produces structured design recommendation       │
│  6. Assesses confidence → escalate? or respond?     │
│  7. Sends A2A reply → a2a_team_leader_inbox         │
└─────────────────┬───────────────────────────────────┘
                  │ MCP (HTTP)
┌─────────────────▼───────────────────────────────────┐
│  Layer 4: MCP Server (Tool Abstraction)             │
│  ospf_parser, interface_parser, network_design,     │
│  connectivity_tester, security_parser               │
└─────────────────┬───────────────────────────────────┘
                  │ SSH
┌─────────────────▼───────────────────────────────────┐
│  Layer 5: Real Cisco Routers                        │
│  192.168.255.10, .20, .30 via Azure EVE-NG          │
└─────────────────────────────────────────────────────┘
```

### What this means for training data

The model must learn:
- **Input format**: A2A JSON task structure
- **Tool-call syntax**: MCP HTTP call format
- **Output format**: Structured A2A reply with confidence, sources, escalation flag
- **Decision logic**: When to call a tool vs. answer from knowledge
- **Escalation logic**: When to flag for human approval

---

## 4. The Three-Stage Training Plan

```
STAGE 1 — SFT: Knowledge + Format            (Week 1, ~12 hours GPU)
┌─────────────────────────────────────────────────────┐
│  9,000 samples                                      │
│  Teaches: CCDE terminology, <think> format,         │
│  A2A task parsing, basic tool-calling syntax        │
│  Method: QLoRA (Unsloth, r=32, 4-bit)              │
│  Expected result: 65%+ keyword match                │
└──────────────────────────┬──────────────────────────┘
                           │
                           ▼
STAGE 2 — GRPO: Reasoning Quality             (Week 2, ~20 hours GPU)
┌─────────────────────────────────────────────────────┐
│  800 verifiable samples                             │
│  Teaches: genuine reasoning on exact calculations,  │
│  correct tool selection, appropriate escalation     │
│  Method: GRPO (8 rollouts, hybrid reward function)  │
│  Expected result: 75%+ overall score                │
└──────────────────────────┬──────────────────────────┘
                           │
                           ▼
STAGE 3 — Agentic SFT: Tool Use + Multi-turn  (Week 3, ~8 hours GPU)
┌─────────────────────────────────────────────────────┐
│  2,000 agentic trajectory samples                   │
│  Teaches: ReAct loops, clarification seeking,       │
│  multi-turn design sessions, A2A full workflow      │
│  Method: SFT on trajectory data                     │
│  Expected result: Fully agentic behaviour           │
└─────────────────────────────────────────────────────┘
```

### Why this order matters

**SFT first** — GRPO needs a model that already produces coherent domain output. Starting GRPO on the base Phi-4 (without domain knowledge) produces incoherent rollouts. This is what happened with early DeepSeek attempts before the SFT cold-start phase was introduced.

**GRPO second** — SFT teaches imitation. GRPO teaches judgement. The model needs to know the domain before it can reason about what's correct.

**Agentic SFT third** — Tool-calling trajectories require the model to already understand the domain deeply. If the model doesn't know what `ospf_parser` data means, it cannot reason about which tool to call next.

---

## 5. Stage 1 — Dataset Design

### Core principle: Quality > Quantity

One high-quality sample (with structured `<think>` steps, exact CCDE terms, cited standards) trains as effectively as 10+ templated low-quality samples.

### Dataset composition: 9,000 samples

```
┌─────────────────────────────────────────────────────────┐
│  TYPE 1: Design scenarios              3,500 (39%)      │
│  TYPE 2: Troubleshooting               2,000 (22%)      │
│  TYPE 3: Protocol deep-dives           1,500 (17%)      │
│  TYPE 4: Compare & select              1,000 (11%)      │
│  TYPE 5: Calculations & capacity         500  (6%)      │
│  TYPE 6: Design methodology              500  (5%)      │
└─────────────────────────────────────────────────────────┘
```

### Type 1: Design scenarios (3,500 samples)

The model's primary job. Each sample must vary across:

```
Variables to combine:
  users:       [50, 200, 500, 1500, 3000, 8000, 20000, 50000]
  sites:       [1, 3, 10, 25, 80, 200]
  compliance:  [none, PCI-DSS, HIPAA, SOX, NIST-CSF, FedRAMP]
  uptime_sla:  [99.0, 99.5, 99.9, 99.95, 99.99, 99.999]
  industry:    [finance, healthcare, retail, education, government, manufacturing]
  wan_type:    [none, MPLS, SD-WAN, internet-only, hybrid]
  dc:          [true, false]
  wireless:    [true, false]
  budget:      [tight, moderate, enterprise, unlimited]
```

Sub-categories and sample counts:
```
Campus topology design          700 samples
  - Collapsed core (< 500 users)
  - Two-tier (500–3000 users)
  - Three-tier (3000–20000 users)
  - Capacity planning calculations

Data center design              600 samples
  - Spine-leaf (east-west traffic)
  - Three-tier (north-south traffic)
  - VXLAN BGP EVPN fabric
  - Multi-DC active-active

WAN / SD-WAN design             600 samples
  - Hub-and-spoke (< 20 branches)
  - Partial mesh (20–80 branches)
  - SD-WAN migration from DMVPN
  - Dual-ISP with SLA policies

Compliance-driven design        700 samples
  - PCI-DSS CDE isolation
  - HIPAA medical device segmentation
  - Multi-framework (PCI + HIPAA)
  - Zero-trust overlay

HA / redundancy design          900 samples
  - SLA calculations (exact numbers)
  - N+1 vs N+2 analysis
  - Active-active vs active-standby
  - FHRP selection and design
```

### Type 2: Troubleshooting (2,000 samples)

These teach OSI-bottom-up diagnostic methodology — critical for the agentic tool-calling stage.

```
BGP troubleshooting             500 samples
  - Neighbor state machine issues
  - Route missing / not advertised
  - AS-path loop prevention (as-override, allowas-in)
  - Route dampening tuning
  - Policy and filter debugging

OSPF troubleshooting            400 samples
  - Neighbor state issues (EXSTART, 2WAY)
  - LSA flooding / high CPU
  - Area type mismatch
  - MTU mismatch (DBD won't exchange)
  - Convergence tuning

MPLS / VPN troubleshooting      300 samples
  - VRF leaking / isolation
  - RD/RT misconfiguration
  - Label distribution issues
  - PE-CE routing problems

Interface / connectivity        400 samples
  - CRC errors → cable / SFP diagnosis
  - Duplex mismatch
  - MTU black holes
  - Asymmetric routing

QoS troubleshooting             400 samples
  - DSCP misclassification
  - Queue drops on WAN
  - Trust boundary issues
  - VoIP quality degradation
```

### Type 3: Protocol deep-dives (1,500 samples)

Teach exact terminology — the primary gap identified in the quality report.

```
VXLAN BGP EVPN                  400 samples
  Must include exact terms:
  - "symmetric IRB" vs "asymmetric IRB"
  - "L3VNI" (per-VRF)
  - "distributed anycast gateway"
  - "BGP EVPN control plane" (RFC 7432)
  - "50-byte VXLAN overhead"
  - "VTEP (Virtual Tunnel Endpoint)"

MPLS VPN                        350 samples
  Must include exact terms:
  - "Route Distinguisher" (makes routes unique in BGP)
  - "Route Target" (controls import/export)
  - "VRF isolation"
  - Inter-AS "Option A/B/C"
  - "ASBR (Autonomous System Border Router)"

SD-WAN Architecture             350 samples
  Must include exact terms:
  - "vBond (orchestration plane)"
  - "vManage (management plane)"
  - "vSmart (control plane, OMP)"
  - "vEdge/cEdge (data plane)"
  - "BFD probes for SLA measurement"
  - "brownout detection"
  - "application-aware routing"

QoS Deep-Dive                   400 samples
  Must include exact terms:
  - "EF (Expedited Forwarding) DSCP 46" for VoIP
  - "LLQ (Low Latency Queuing)"
  - "CBWFQ (Class-Based Weighted Fair Queuing)"
  - "33% priority queue limit"
  - "trust boundary at access layer"
  - "traffic shaping buffers, policing drops"
  - "TCP retransmission behavior"
```

### Type 4: Compare & select (1,000 samples)

Critical for design decisions — the model must justify technology choices.

```
Examples:
- HSRP vs VRRP vs GLBP
  → "GLBP active-active", "AVG/AVF roles", "HSRP preemption risks"
- Spine-leaf vs three-tier for DC
  → "east-west vs north-south", "VM mobility", "consistent latency"
- OSPF vs BGP as campus IGP
  → "OSPF for single-vendor campus", "topology hiding"
- Hub-and-spoke vs full mesh
  → "N(N-1)/2 formula", "partial mesh compromise"
- GPTQ vs GGUF for inference
  → meta-lesson: model knows its own deployment options
```

### Type 5: Calculations & capacity (500 samples)

These are the GRPO seed data — exact, verifiable answers.

```
SLA downtime calculations:
  99.9%  → 8.77 hours/year, 43.8 min/month
  99.99% → 52.6 minutes/year, 4.38 min/month  ← most common
  99.999% → 5.26 minutes/year

Spine-leaf capacity:
  64-port switches, 32 uplinks: 32 leaf × 32 servers = 1,024 servers
  2-tier formula: (N_leaf × downlink_ports) = total servers
  N×M/2 bisection bandwidth formula

WAN link count:
  Hub-and-spoke: N links (N = branch count)
  Full mesh: N(N-1)/2 links
  80 branches full mesh: 80×79/2 = 3,160 links

BGP timers:
  Default keepalive: 60s, hold-time: 180s
  Fast convergence: keepalive 3s, hold-time 9s
  BFD sub-second: 300ms detection

VXLAN overhead:
  VXLAN header: 8 bytes
  UDP: 8 bytes
  Outer IP: 20 bytes
  Outer Ethernet: 14 bytes (+ 4 FCS)
  Total: 50 bytes overhead
  Minimum MTU: 1500 + 50 = 1550 bytes
```

### Type 6: Design methodology (500 samples)

Teaches the CCDE top-down approach — most missed in the quality report.

```
Key concepts every methodology sample must include:
  1. "top-down design" (business → applications → network)
  2. "business requirements" (SLA, growth, cost constraints)
  3. "technical requirements" (bandwidth, latency, redundancy)
  4. "organizational constraints" (budget, team skills, vendor preference)
  5. "application criticality" (tier 1/2/3 classification)

Sample question types:
  - Requirements gathering from a vague customer brief
  - Converting business SLAs to network parameters
  - Prioritising competing design goals
  - Justifying technology selection to a non-technical stakeholder
  - Total Cost of Ownership calculations
```

### Sample quality standard

Every single sample in the dataset must meet this bar:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a CCDE-level network architect..."
    },
    {
      "role": "user",
      "content": "We need 99.99% uptime for our trading platform data center. Compare N+1 vs N+2 redundancy. What's the maximum allowed downtime and what capacity rule applies for active-active?"
    },
    {
      "role": "assistant",
      "content": "<think>\nStep 1: Calculate downtime budget.\n99.99% uptime = 0.01% downtime per year\n365.25 days × 24 hours × 60 min × 0.0001 = 52.6 minutes/year\n4.38 minutes/month\n26.3 seconds/week\n\nStep 2: N+1 vs N+2 analysis.\nN+1: one redundant component beyond minimum. Cost-optimal, handles single failure.\nN+2: two redundant components. Handles simultaneous dual failure — diminishing returns and convergence complexity increases.\nFor 99.99%: N+1 is the correct choice. N+2 adds cost and MTTF complexity without SLA improvement unless MTTR > 4.38 min.\n\nStep 3: Active-active capacity rule.\nEach site must be sized to carry 100% of load — not 50%.\nIf Site A fails, Site B takes full traffic. Sizing at 50% causes cascade failure.\nRule: each node/site must handle N×100% (not N×50%).\n\nStep 4: GSLB requirement.\nActive-active DC requires GSLB (Global Server Load Balancing) for application-layer failover.\nDNS TTL must be ≤ 30 seconds for sub-minute failover.\n</think>\n\n**Uptime Budget:** 99.99% SLA = **52.6 minutes of allowed downtime per year** (4.38 min/month).\n\n**N+1 is correct for 99.99%.** N+2 provides marginal SLA improvement but significantly increases:\n- Capital cost (additional redundant components)\n- Convergence complexity (more failure scenarios to test)\n- MTTF calculations become harder to validate\n\nN+2 is only justified when MTTR consistently exceeds the SLA budget — rare in modern DC environments with automated failover.\n\n**Active-active capacity rule: each site must be sized for 100% of peak load.** Designing at 50% causes a cascade failure the moment one site goes down.\n\n**Implementation:** GSLB with DNS TTL ≤ 30s, BFD for sub-second failure detection, automated runbook for failover validation.\n\nSources: [Campus Network HA Design Guide], [Designing for Cisco Network Service Architectures ARCH], RFC 5880 (BFD)"
    }
  ]
}
```

**Checklist for every sample:**
- [ ] `<think>` block with numbered steps
- [ ] At least 1 exact number or calculation
- [ ] At least 2 CCDE-specific terms (exact phrasing)
- [ ] At least 1 cited source (RFC, Cisco CVD, or book title in brackets)
- [ ] Answer is at least 150 words (concise but complete)
- [ ] No hallucinated product names or version numbers

---

## 6. Stage 1 — SFT Training Strategy

### Method: QLoRA via Unsloth

**Why QLoRA (not full fine-tuning):**

| | Full fine-tuning | QLoRA |
|--|-----------------|-------|
| Updates | All 14B parameters | ~50M LoRA adapter params |
| VRAM needed | ~160 GB (8× A100) | ~9 GB (1× T4) |
| Training time | ~80 hours | ~12 hours |
| Cost (AWS) | ~$240 | ~$6 |
| Catastrophic forgetting risk | High | Low (base frozen) |
| Suitable for domain adaptation | Overkill | ✅ Correct tool |

**Why Unsloth (not vanilla HuggingFace):**
- 2× faster training via custom CUDA kernels for attention
- 60% less VRAM via smart gradient checkpointing
- Drop-in replacement for HF Trainer — no code changes needed

### LoRA hyperparameter decisions

```yaml
r: 32
# The adapter rank. Controls capacity.
# r=8:  fast, learns format, misses deep CCDE knowledge
# r=16: good for instruction following, weak on technical depth
# r=32: correct for domain adaptation with 10k samples  ← CHOSEN
# r=64: risks overfitting on our dataset size, uses 2× VRAM
# r=128: for massive datasets (100k+), not needed here

lora_alpha: 32
# Scale factor = alpha/r = 1.0
# Standard practice: alpha = r gives stable training
# Higher alpha/r ratio (e.g. alpha=64, r=32) = aggressive adaptation
# We keep 1.0 because the model already has the base knowledge

lora_dropout: 0.05
# Small regularisation. Higher dropout (0.1) can help with small datasets
# but 9k samples with high-quality data doesn't need aggressive regularisation

target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
# All 7 projection types. More coverage = better domain adaptation
# Minimum viable: [q_proj, v_proj] — attention only
# We target all because we need both attention and MLP adaptation

use_rslora: false
# RSLoRA normalises by sqrt(r) instead of r. Better for high r (64+)
# Not needed at r=32
```

### Training hyperparameter decisions

```yaml
learning_rate: 1.0e-4
# Standard for LoRA. Key insight: LR that works for full fine-tuning
# needs to be 10-100× higher for LoRA because only ~0.3% of params are updating.
# Too low (1e-5): model barely adapts — wastes compute
# Too high (5e-4): loss spikes, unstable training, LoRA diverges
# 1e-4 is the empirical sweet spot for QLoRA domain adaptation

lr_scheduler: cosine
# Cosine decay is smooth. Linear is also fine.
# Cosine prevents catastrophic forgetting of format near end of training
# (learning rate drops smoothly, model doesn't "unlearn" reasoning format)

warmup_steps: 100
# 100 steps at near-zero LR before ramping up
# Prevents large gradient updates from disturbing the base model
# at the start when LoRA weights are random

num_train_epochs: 3
# 3 epochs on 9k samples = 27k gradient steps
# Epoch 1: model learns the format and basic terminology
# Epoch 2: model learns domain-specific patterns
# Epoch 3: model consolidates — monitor for overfitting here

per_device_batch_size: 1
gradient_accumulation_steps: 16
# Effective batch size = 1 × 16 = 16
# Can't do batch_size > 1 with 14B model on T4 (16 GB)
# Gradient accumulation simulates larger batches

optim: adamw_8bit
# 8-bit Adam: saves ~4 GB VRAM vs standard Adam
# Uses blocked quantization for the optimizer state
# No quality difference from standard Adam for LoRA

bf16: true
# bfloat16 on Ampere+ GPUs (T4 supports it)
# bf16 vs fp16: bf16 has larger dynamic range, fewer overflow issues
# Critical for training stability with large models
```

### The reasoning-weighted loss — why it matters

Standard cross-entropy loss treats all tokens equally. But for our use case:

```
<think>
Step 1: Analyze the network scale — 8,000 users single building
Step 2: Three-tier is indicated above 3,000 users
</think>
Three-tier campus architecture is recommended.
```

The answer (`Three-tier campus...`) is 5 tokens.
The reasoning (`Step 1... Step 2...`) is 40 tokens.

Without weighting: the model learns mostly from the short answer (more gradient from shorter, repeating patterns).

With 2× weighting on think tokens: the model learns that **the quality of the reasoning chain matters as much as the final answer**. This is what produces genuine CCDE-depth analysis rather than shallow one-liners.

```python
# Implementation: per-token weight tensor
# Tokens inside <think>...</think>  → weight = 2.0
# All other tokens                  → weight = 1.0
# Applied during loss.backward() as a multiplier on the cross-entropy
```

### What to monitor during training

```
Good signs:
  ✅ train_loss dropping smoothly (2.5 → 1.8 → 1.4 over 3 epochs)
  ✅ eval_loss tracking within 0.2 of train_loss (no overfitting)
  ✅ <think> blocks appearing in sample completions by epoch 2
  ✅ CCDE terms appearing in completions by epoch 1

Warning signs:
  ⚠  eval_loss rising while train_loss drops → overfitting
     Fix: reduce epochs to 2, add lora_dropout=0.10
  ⚠  loss plateaus above 2.0 after 500 steps
     Fix: increase learning_rate to 2e-4
  ⚠  model stops producing <think> blocks mid-training
     Fix: check think token IDs, verify reasoning-weighted loss is active
  ⚠  GPU OOM
     Fix: pipeline doubles gradient_accumulation_steps automatically
```

---

## 7. Stage 2 — GRPO for Reasoning

### What GRPO teaches that SFT cannot

SFT teaches the model to **imitate** good answers. The model learns patterns from the training data.

GRPO teaches the model to **discover** correct answers through exploration and reward. The model learns which reasoning strategies actually produce correct outcomes.

**The critical difference on a real question:**

```
Question: "What is the maximum allowed downtime for 99.99% uptime?"

SFT model: Produces "52.6 minutes per year" IF it saw similar examples.
           Produces "about 1 hour" if the training data was imprecise.

GRPO model: Generates 8 different answers, discovers that "52.6 minutes"
            scores highest (reward function verifies the calculation),
            reinforces the precise calculation path.
```

### The reward function — most important design decision

GRPO lives or dies by the reward function. Here is ours:

```python
def reward_fn(response: str, question: dict) -> float:
    score = 0.0

    # ── Component 1: Format reward (0.0 – 0.25) ──────────────
    # Did the model use the <think> reasoning format?
    if "<think>" in response and "</think>" in response:
        score += 0.15
    # Did it use numbered steps?
    step_count = len(re.findall(r"Step\s+\d+", response))
    score += min(step_count * 0.025, 0.10)   # up to 0.10 for 4+ steps

    # ── Component 2: Exact answer reward (0.0 – 0.50) ────────
    # For verifiable questions only — checked against answer_key
    expected_terms = question.get("expected_key_facts", [])
    if expected_terms:
        hits = sum(1 for t in expected_terms if t.lower() in response.lower())
        score += 0.50 * (hits / len(expected_terms))

    # ── Component 3: Calculation accuracy (0.0 – 0.20) ───────
    # For numeric questions — check exact numbers appear
    exact_numbers = question.get("exact_values", [])
    for num in exact_numbers:
        if str(num) in response:
            score += 0.20 / max(len(exact_numbers), 1)

    # ── Component 4: Source citation reward (0.0 – 0.05) ─────
    if re.search(r"RFC\s+\d{4}", response) or \
       re.search(r"\[.{4,60}\]", response):
        score += 0.05

    return min(score, 1.0)
```

### GRPO dataset: 800 verifiable samples

Only questions with **objectively checkable answers** qualify for GRPO:

```
Calculations (exact numbers)          200 samples
  - SLA downtime budgets
  - Spine-leaf capacity formulas
  - WAN link count formulas
  - VXLAN MTU calculations
  - BGP timer math

Protocol specifics (exact terms)      300 samples
  - IRB mode names (symmetric/asymmetric)
  - MPLS Inter-AS option names
  - QoS DSCP values
  - OSPF LSA type descriptions
  - SD-WAN plane names

Comparative analysis (ranked choices) 200 samples
  - Technology A vs B with clear winner
  - Option ranking (most secure, most scalable)

Escalation decisions (binary)         100 samples
  - Given this scenario, should the agent escalate? Yes/No + reason
  - PCI-DSS compliance present → always YES
  - 99.99% uptime → always YES
  - Routine campus question → always NO
```

### GRPO hyperparameters

```yaml
num_generations: 8
# Generate 8 candidate responses per question.
# More = better signal, but 8× the inference cost.
# 8 is the DeepSeek-R1 default and works well.

learning_rate: 5.0e-6
# 20× lower than SFT. GRPO makes larger policy updates —
# too high a LR collapses the policy (all outputs become identical).

per_device_batch_size: 1
gradient_accumulation_steps: 8
# GRPO needs less accumulation than SFT — each step has
# 8 rollouts already providing gradient diversity.

kl_coeff: 0.04
# KL divergence penalty against the reference model.
# Prevents the model from drifting too far from the SFT checkpoint.
# Too low: model exploits reward function (reward hacking)
# Too high: model barely updates

max_new_tokens: 2048
# Must be long enough to allow full reasoning chains.
# GRPO naturally learns to use the full budget when it improves rewards.
```

---

## 8. Stage 3 — Agentic Behaviour Training

This is what makes the difference between a question-answering model and an autonomous agent.

### The four agentic behaviours to train

**Behaviour 1: Tool call decision**
The model must learn WHEN to call a tool vs. answer from internal knowledge.

```
Rule: Call a tool when you need CURRENT STATE of a live network.
      Answer from knowledge when the question is about DESIGN PRINCIPLES.

Call ospf_parser WHEN:  "Is OSPF healthy on router-1 right now?"
Answer from knowledge:  "How should I design OSPF areas for 500 routers?"
```

Training data format:
```json
{
  "messages": [
    {"role": "system", "content": "...tools: ospf_parser, interface_parser, connectivity_tester..."},
    {"role": "user",   "content": "Router-1 in our campus is showing slow convergence. What's wrong?"},
    {"role": "assistant", "content": "<think>\nThis requires current OSPF state — I need to call ospf_parser\nbefore I can diagnose. Answering from knowledge alone would be\nguesswork without seeing actual neighbor states and LSA counts.\n</think>\n<tool_call>\n{\"tool\": \"ospf_parser\", \"params\": {\"device_id\": \"router-1\", \"idempotency_key\": \"ospf-router1-diag-001\"}}\n</tool_call>"},
    {"role": "tool",      "content": "{\"neighbors\": [{\"id\": \"2.2.2.2\", \"state\": \"FULL\"}, {\"id\": \"3.3.3.3\", \"state\": \"2WAY\"}], \"lsa_count\": 4823}"},
    {"role": "assistant", "content": "<think>\nNeighbor 3.3.3.3 is stuck in 2WAY — not reaching FULL. This means\nthe DBD exchange is failing. 4,823 LSAs is very high — likely\na flat area problem contributing to slow SPF calculation.\n\nRoot causes to investigate:\n1. MTU mismatch with 3.3.3.3 (prevents DBD completion)\n2. Area type mismatch\n3. Authentication mismatch\n</think>\n\nFound the issue: neighbor **3.3.3.3 is stuck in 2WAY** state (not reaching FULL adjacency). Combined with 4,823 LSAs suggesting a flat area problem.\n\n**Immediate diagnosis steps:**\n1. `show ip ospf neighbor 3.3.3.3 detail` — check MTU mismatch\n2. `show ip ospf interface Gi0/1` — verify area type matches\n3. `debug ip ospf adj` — confirm DBD exchange failure reason\n\n**Likely fix:** MTU mismatch — add `ip ospf mtu-ignore` or fix interface MTU to match."}
  ]
}
```

**Behaviour 2: Clarification seeking**
The model must ask for missing requirements rather than guessing.

```json
{
  "messages": [
    {"role": "user",      "content": "Design me a network for our new office."},
    {"role": "assistant", "content": "<think>\nThe request is too vague to produce a useful design.\nKey unknowns: user count, site count, compliance, uptime SLA, budget.\nI should ask for requirements before designing — guessing would\nproduce a design that may be completely wrong for their needs.\n</think>\n\nBefore I can design your network, I need to understand your requirements:\n\n1. **Scale:** How many users? Any plans to grow in 2–3 years?\n2. **Sites:** Is this a single office or multiple locations?\n3. **Compliance:** Any regulatory requirements? (PCI-DSS, HIPAA, SOX?)\n4. **Uptime SLA:** What level of availability is required? (99.9% = 8.7 hr/year downtime, 99.99% = 52.6 min/year)\n5. **Budget:** Rough range? (Collapses the option space significantly)\n6. **Applications:** Any latency-sensitive apps? (VoIP, video, trading?)\n\nWith these answers I can give you a precise architecture recommendation."}
  ]
}
```

**Behaviour 3: Escalation decisions**
The model must know its own boundaries and flag appropriately.

```json
{
  "messages": [
    {"role": "user", "content": "Design the network for our new PCI-DSS compliant payment processing centre, 500 staff, 99.99% uptime."},
    {"role": "assistant", "content": "<think>\nThis request has two automatic escalation triggers:\n1. PCI-DSS compliance → always requires human review per policy\n2. 99.99% uptime SLA → critical infrastructure design\n\nI will produce the design but MUST flag it for human approval\nbefore any implementation begins. A QSA (Qualified Security\nAssessor) must review the PCI-DSS segmentation.\n</think>\n\n**⚠️ REQUIRES HUMAN APPROVAL** before implementation:\n- PCI-DSS compliance design requires QSA review\n- 99.99% uptime is critical infrastructure (52.6 min/year budget)\n\n---\n\n**Architecture recommendation:**\n[Full design here...]\n\n---\n**Escalation reason:** `compliance=[PCI-DSS], uptime=99.99%`\n**Confidence:** 0.91\n**Recommended reviewer:** Network Security Architect + QSA"}
  ]
}
```

**Behaviour 4: A2A task format awareness**
The model must understand it receives structured JSON tasks and must reply in A2A format.

```json
{
  "messages": [
    {"role": "system",    "content": "You are the Virtual Architect agent in a multi-agent system. You receive tasks via A2A protocol and must reply in A2A format."},
    {"role": "user",      "content": "{\"message_type\": \"task_delegation\", \"task_id\": \"task-abc123\", \"from_agent\": \"team_leader\", \"payload\": {\"type\": \"design\", \"context\": {\"requirements\": {\"users\": 3000, \"sites\": 5, \"uptime\": 99.99, \"compliance\": [\"PCI-DSS\"]}}}}"},
    {"role": "assistant", "content": "{\"message_type\": \"task_response\", \"task_id\": \"task-abc123\", \"from_agent\": \"virtual_architect\", \"to_agent\": \"team_leader\", \"status\": \"success\", \"payload\": {\"reasoning\": \"Step 1: Scale analysis...\", \"recommendation\": \"Three-tier campus with PCI-DSS CDE isolation...\", \"confidence\": 0.88, \"sources\": [\"CCNP Enterprise Design ENSLD\", \"PCI-DSS 4.0.1 Req 1.3\"], \"requires_human_approval\": true, \"escalation_reason\": \"PCI-DSS compliance + 99.99% uptime SLA\"}}"}
  ]
}
```

### Agentic dataset: 2,000 samples

```
Tool-calling trajectories (ReAct loops)    800 samples
  - Single tool call + observe + answer
  - Multi-tool chains (ospf → interface → connectivity)
  - Tool call → bad result → different tool

Clarification dialogs                      400 samples
  - Vague request → questions → requirements → design
  - Partial requirements → targeted questions
  - Complete requirements → no questions, proceed directly

Escalation decisions                       300 samples
  - PCI-DSS present → escalate (always)
  - HIPAA present → escalate (always)
  - 99.99% uptime → escalate (always)
  - Confidence < 0.80 → escalate
  - Routine question → do not escalate

A2A format I/O                             300 samples
  - Receive JSON task → produce JSON response
  - Various task types (design, troubleshoot, estimate)
  - Error handling (malformed task → graceful error response)

Multi-turn design sessions                 200 samples
  - Turn 1: gather requirements
  - Turn 2: propose high-level design
  - Turn 3: answer follow-up ("what about the DR site?")
  - Turn 4: refine design
```

---

## 9. Evaluation Framework

### Automated scoring (runs after each training stage)

```bash
python test_model_quality.py --suite --save --backend api
```

Scoring formula per test case:
```
overall_score = (0.50 × keyword_match)
              + (0.20 × source_citation)
              + (0.20 × confidence_heuristic)
              + (0.10 × think_block_bonus)
```

Pass thresholds:
- **PASS:** ≥ 0.70
- **WARN:** 0.45–0.69
- **FAIL:** < 0.45

### Target scores per stage

| Metric | Baseline (GPT-4o) | After Stage 1 SFT | After Stage 2 GRPO | After Stage 3 Agentic |
|--------|:-----------------:|:-----------------:|:-----------------:|:---------------------:|
| Overall score | 44% | 65% | 75% | 80%+ |
| Keyword match | 13% | 60% | 72% | 72% |
| `<think>` rate | 96%* | 98% | 99% | 99% |
| PASS count | 1/25 | 14/25 | 18/25 | 20/25 |
| Tool use correct | N/A | N/A | N/A | 85%+ |
| Escalation correct | N/A | N/A | N/A | 95%+ |

*GPT-4o `<think>` rate is forced by system prompt, not trained behaviour

### Human evaluation rubric (for the 5 hardest cases)

For cases tc_009 (VXLAN EVPN), tc_014 (Inter-AS MPLS), tc_007 (RD/RT), tc_004 (OSPF redesign), tc_015 (methodology):

```
Score 1–5 on each dimension:

Technical accuracy:     Is the design technically correct?
CCDE depth:             Does it address the problem at CCDE level (not CCNA)?
Reasoning quality:      Are the <think> steps logical and complete?
Practical applicability: Could a real engineer implement this?
Source quality:          Are cited sources authoritative and relevant?
```

---

## 10. Deployment Strategy

### Quantization: What and why

After Stage 3 training, two quantized versions are produced:

**GPTQ (GPU deployment):**
```
14B parameters × 4-bit = ~7 GB on disk
GPU VRAM at inference: ~9 GB (model + KV cache)
Inference speed: 3–6 seconds per request
Target hardware: T4, A10G, RTX 3090/4090
```

**GGUF Q4_K_M (CPU deployment / edge):**
```
14B parameters × 4-bit (K-quant = mixed precision) = ~8.7 GB
RAM at inference: ~12 GB
Inference speed: 15–30 seconds per request
Target hardware: Any server with 16+ GB RAM
Use case: No GPU available, dev testing, edge deployment
```

### Serving architecture

```
Option A — Standalone (demo / development):
  uvicorn api.main:app --port 8000

Option B — Dream Team integrated:
  python dream_team_integration/virtual_architect_agent.py
  (A2A event loop, connects to Redis, waits for tasks)

Option C — Docker (production):
  docker-compose up -d
  (phi4-api + phi4-mcp + nginx, GPU via nvidia-container-toolkit)

Option D — systemd (bare metal EC2):
  systemctl start phi4-api
  (auto-restart, log rotation, firewall via nginx)
```

### A2A integration checklist

Before deploying to the Dream Team:

```
[ ] Redis is running (docker-compose includes it)
[ ] REDIS_URL environment variable is set
[ ] Model is loaded and /health returns model_loaded: true
[ ] test_model_quality.py --suite passes ≥ 18/25 test cases
[ ] A2A inbox is configured: a2a_virtual_architect_inbox
[ ] Team Leader routes "design" type tasks to virtual_architect
[ ] Approval queue is connected to Streamlit UI
[ ] Confidence threshold 0.80 is tuned for false positive rate
[ ] PCI-DSS and HIPAA escalation rules are tested
```

---

## 11. Key Decisions Log

| Decision | Choice | Alternatives considered | Rationale |
|----------|--------|------------------------|-----------|
| Base model | Phi-4-14B | Llama-3.1-8B, Qwen2.5-14B | Native `<think>` tokens, strong technical knowledge, fits T4 GPU |
| Fine-tuning method | QLoRA | Full FT, LoRA (no quant), PEFT adapter | Cost/VRAM balance; domain adaptation, not capability building |
| LoRA rank | r=32 | r=8, r=16, r=64 | Sufficient capacity for 9k samples without overfitting |
| Dataset size | 9,000 SFT + 800 GRPO + 2,000 agentic | 2k, 5k, 20k | Coverage math: ~420 CCDE scenarios × 20 variations |
| Reasoning loss | 2× weight on `<think>` tokens | Standard cross-entropy | Forces model to treat reasoning as important as the answer |
| GRPO strategy | After SFT (two-stage) | GRPO only, DPO | SFT provides cold start; GRPO can't learn from incoherent rollouts |
| Agentic training | Separate Stage 3 | Interleaved with SFT | Model needs domain knowledge before it can make tool decisions |
| Quantization | GPTQ (GPU) + GGUF (CPU) | FP16 only, AWQ | Deployment flexibility; GPTQ for production, GGUF for dev |
| Escalation threshold | confidence < 0.80 | 0.70, 0.90 | Validated against test suite false positive rate |
| Effective batch size | 16 (1×16 accum) | 8, 32 | T4 memory limit forces batch=1; 16 accumulation steps is standard |

---

## 12. Baseline Results

Tested against 25 CCDE test cases using GPT-4o-mini (as baseline before fine-tuning).
Full results in `QUALITY_REPORT.md` and `quality_report.json`.

### Summary

| Metric | Score |
|--------|------:|
| Overall score | 44% |
| Keyword match | 13% |
| Confidence | 65% |
| `<think>` block rate | 96% (forced by system prompt) |
| PASS / WARN / FAIL | 1 / 7 / 17 |

### Top training gaps identified

| Domain | Missing terms | Priority |
|--------|---------------|----------|
| QoS | `EF DSCP 46`, `LLQ`, `33% priority queue limit`, `trust boundary` | HIGH |
| VXLAN EVPN | `symmetric IRB`, `L3VNI`, `distributed anycast gateway` | HIGH |
| OSPF areas | `totally stubby`, `type 3 LSA`, `ABR summarization` | HIGH |
| BGP edge cases | `AS-override`, `allowas-in`, dampening mechanics | HIGH |
| MPLS VPN | Inter-AS Options A/B/C, `ASBR requirements` | HIGH |
| HA maths | `52.6 min/year`, `N+1 optimal`, `50% utilisation rule` | MEDIUM |
| SD-WAN | `vBond/vManage/vSmart/vEdge`, `BFD SLA`, `brownout detection` | MEDIUM |
| Methodology | `top-down design`, `business requirements`, all 5 constraint types | MEDIUM |

---

## 13. Course Exercises

### Exercise 1 — Understand the agentic loop

**Task:** Trace a single A2A task through the full system:
1. Trigger a test incident: `curl -X POST http://localhost:8001/incident/trigger`
2. Watch the Team Leader logs: `docker-compose logs -f team-leader`
3. Watch the Virtual Architect logs: `docker-compose logs -f virtual-architect`
4. Observe the Redis queue: `redis-cli LLEN a2a_virtual_architect_inbox`
5. Find the response in the UI at `http://localhost:8501`

**Question:** At which step does the Virtual Architect decide whether to escalate to human approval?

---

### Exercise 2 — Dataset quality review

**Task:** Review 10 samples from the generated dataset:
```bash
python -c "
import json
with open('generated_data/training_data.jsonl') as f:
    for i, line in enumerate(f):
        if i >= 10: break
        s = json.loads(line)
        print(s['messages'][-1]['content'][:500])
        print('---')
"
```

**Checklist for each sample:**
- [ ] Does it have a `<think>` block?
- [ ] Does the `<think>` block have numbered steps?
- [ ] Does it use exact CCDE terminology?
- [ ] Does it cite at least one source in brackets?

**Question:** What percentage of your generated samples meet all four criteria?

---

### Exercise 3 — LoRA rank experiment

**Task:** Train two small models (500 samples, 1 epoch) with different ranks:
```bash
# Model A: r=8
python fine_tuning/train.py --max-samples 500 \
  --config fine_tuning/config_r8.yaml

# Model B: r=32
python fine_tuning/train.py --max-samples 500 \
  --config fine_tuning/config.yaml
```

**Evaluate both:**
```bash
python test_model_quality.py --suite --backend api --save
```

**Question:** At what point does higher rank stop improving keyword match? Why?

---

### Exercise 4 — Design your reward function

**Task:** Write a reward function for this question:
> "What is the VXLAN header overhead and what MTU is required on the underlay?"

The correct answer includes: `50 bytes overhead`, `MTU 1550 minimum`, `VTEP`

```python
def reward_vxlan_mtu(response: str) -> float:
    # Your implementation here
    pass
```

**Question:** What happens to training if your reward function has a bug that rewards longer responses regardless of correctness?

---

### Exercise 5 — Agentic trajectory construction

**Task:** Write one complete agentic training sample for this scenario:

> The Team Leader sends a design task: 3,000 users, 3 sites, no compliance,
> 99.9% uptime. But the agent notices the current OSPF state on the core
> routers shows 4 of 6 neighbors are DOWN.

Your sample must include:
1. The A2A task JSON (input)
2. The `<think>` block deciding to call ospf_parser
3. The tool call
4. The tool result (mock data showing 4/6 neighbors down)
5. The `<think>` block integrating both the design request and the live state
6. The final A2A response noting the live network issue affects the design recommendation

---

## Appendix: File Reference

```
fine_tuned_llms/Phi-4-14B/
├── FINETUNING_COURSE.md          ← This document
├── QUALITY_REPORT.md             ← Baseline evaluation results
├── quality_report.json           ← Machine-readable baseline
├── test_model_quality.py         ← Evaluation runner
│
├── data_generation/
│   ├── dataset_generator.py      ← Stage 1 & 3 data generation
│   ├── reasoning_chain_builder.py← <think> chain construction helper
│   └── split.py                  ← Train/val/test split
│
├── fine_tuning/
│   ├── train.py                  ← Stage 1 SFT training
│   └── config.yaml               ← Hyperparameters
│
├── evaluation/
│   ├── run_all.py                ← Full eval pipeline
│   ├── technical_accuracy.py     ← Keyword scoring
│   ├── llm_judge.py              ← LLM-as-judge scoring
│   └── test_cases.jsonl          ← 25 CCDE test cases
│
├── dream_team_integration/
│   ├── virtual_architect_agent.py← A2A event loop
│   ├── phi4_inference.py         ← Inference engine (GPTQ/GGUF/HF)
│   ├── mcp_server.py             ← MCP tool server
│   └── config.yaml               ← Agent configuration
│
├── api/
│   ├── main.py                   ← FastAPI server
│   ├── routes/                   ← /design /troubleshoot /estimate
│   └── middleware/               ← Auth + rate limiting
│
├── aws_deployment/
│   ├── launch_ec2.sh             ← Provision GPU instance
│   ├── setup_instance.sh         ← Configure CUDA + Python
│   └── terraform/                ← IaC alternative
│
└── run_full_pipeline.sh          ← 8-stage automated pipeline
```
