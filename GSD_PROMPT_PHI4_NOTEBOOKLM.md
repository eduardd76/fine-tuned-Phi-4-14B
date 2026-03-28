# GSD: Build Phi-4 Network Architect with NotebookLM Knowledge Base

## TL;DR
Build production-ready Phi-4 reasoning model for network architecture. Extract knowledge from NotebookLM (https://notebooklm.google.com/notebook/86b1a8b9-8a9c-486e-8fb3-2e2e0f914528), generate 10k+ training samples with `<think>` tags, fine-tune, deploy. Target: >95% accuracy, <3s inference, book-grounded reasoning.

---

## INPUT
- NotebookLM: https://notebooklm.google.com/notebook/86b1a8b9-8a9c-486e-8fb3-2e2e0f914528
  - Network design books (CCDE-level)
  - Vendor guides (Cisco, Juniper, Arista)
  - Troubleshooting methodologies
  - Compliance guides (PCI-DSS, HIPAA, SOX)
  - Modern tech (SD-WAN, SASE, VXLAN/EVPN)

- Base Model: `microsoft/phi-4` (14B, reasoning-optimized)
- Hardware: RTX 4090 (24GB) or equivalent
- Timeline: 18 hours total

---

## OUTPUT
- Fine-tuned Phi-4: Network architect with book-validated knowledge
- 50,000+ training samples: Every claim grounded in NotebookLM
- Quantized models: GGUF (CPU) + GPTQ (GPU)
- Validation: >95% accuracy vs books, >4.5/5.0 reasoning quality
- Deployment: Production-ready inference with source attribution

---

## DELIVERABLES

### 1. Knowledge Extraction Pipeline
```
knowledge_extraction/
├── extract.py              # Query NotebookLM → structured JSON
├── design_patterns.json    # Topologies, decision criteria, sizing
├── troubleshooting.json    # Diagnostic trees, methodologies
├── compliance.json         # PCI-DSS/HIPAA/SOX specifics
├── vendor_configs.json     # Cisco/Juniper/Arista patterns
└── cost_benchmarks.json    # CapEx/OpEx, timelines, staffing
```

**Queries to run (extract ALL matching content):**
```python
QUERIES = {
    "design": [
        "network topology design patterns and selection criteria",
        "user count ranges for collapsed core vs three-tier architecture",
        "BGP vs OSPF vs EIGRP selection criteria",
        "high availability design patterns and redundancy",
        "SD-WAN and SASE design best practices",
    ],
    "troubleshooting": [
        "BGP troubleshooting methodology step by step",
        "packet loss diagnostic procedures",
        "routing loop causes and fixes",
        "QoS troubleshooting approach",
        "Layer 1-7 network diagnostic sequence",
    ],
    "compliance": [
        "PCI-DSS network segmentation requirements with versions",
        "HIPAA encryption requirements and standards",
        "SOX audit logging requirements for networks",
    ],
    "config": [
        "BGP configuration best practices with examples",
        "ACL structures for compliance",
        "HSRP VRRP configuration patterns",
        "QoS policy examples",
    ],
    "cost": [
        "CapEx costs for enterprise networks by user count",
        "OpEx percentages and breakdown",
        "staffing requirements by network size",
        "implementation timelines by project scale",
    ]
}
```

**Output format (JSON):**
```json
{
  "source_book": "Enterprise Network Design, Chapter 5",
  "pattern_name": "Three-Tier Architecture",
  "criteria": {
    "user_range": "2500-50000",
    "justification": "Provides fault isolation and horizontal scaling",
    "when_to_use": "Enterprise scale, growth expected, budget flexible"
  }
}
```

### 2. Dataset Generator (10,000+ Samples)
```
data_generation/
├── generate.py             # Main: NotebookLM → training samples
├── reasoning_builder.py    # Creates <think> chains from book knowledge
├── validator.py            # Checks technical accuracy vs books
└── config.yaml             # Sample counts, quality thresholds
```

**Sample structure (CRITICAL - all must follow this):**
```json
{
  "messages": [
    {
      "role": "user",
      "content": "Design network for 5000 users, 20 sites, PCI-DSS, 99.99% uptime"
    },
    {
      "role": "assistant",
      "content": "<think>\nStep 1: Topology selection [Source: Enterprise Design, Ch3, p87]\n- 5000 users → three-tier required (collapsed core max 2500)\n- Justification: Scalability to 50k, fault isolation\n\nStep 2: PCI-DSS [Source: Compliance Guide, PCI-DSS 4.0.1 §1.2]\n- CDE segmentation mandatory\n- Dedicated firewall required\n- No direct internet to CDE\n\nStep 3: HA design [Source: Reliability Handbook, Ch8]\n- 99.99% = 52.6min/year max downtime\n- N+1 core minimum, N+2 recommended\n- Active-active data centers\n\n[5-15 steps total, all sourced]\n</think>\n\nDesign: Three-tier architecture...\n[Full design with source attribution]"
    }
  ],
  "metadata": {
    "sources": ["Enterprise Design Ch3", "PCI-DSS 4.0.1", "Reliability Ch8"],
    "patterns": ["three_tier", "pci_segmentation", "active_active_ha"],
    "reasoning_steps": 9
  }
}
```

**Distribution:**
- 7,000 design scenarios (70%)
  - Enterprise campus (2000)
  - Data center (1500)
  - SD-WAN (1000)
  - Cloud connectivity (1000)
  - Security architecture (1500)
  
- 3,000 troubleshooting (30%)
  - Routing issues (1000)
  - Switching/VLAN (500)
  - Performance/QoS (500)
  - Security/firewall (500)
  - HA/redundancy (500)

**Quality gates:**
- ✅ Min 5 NotebookLM citations per sample
- ✅ 100% technical accuracy vs extracted knowledge
- ✅ 5-15 reasoning steps in `<think>` tags
- ✅ Valid config syntax when included
- ✅ Exact compliance versions (e.g., PCI-DSS 4.0.1 §X.Y)

### 3. Fine-Tuning Pipeline
```
fine_tuning/
├── train.py                # Unsloth + Phi-4 training
├── config.yaml             # Hyperparameters
└── requirements.txt
```

**Training config:**
```python
MODEL = "microsoft/phi-4"
MAX_SEQ_LEN = 4096
LORA_RANK = 32
LORA_ALPHA = 32
BATCH_SIZE = 1
GRAD_ACCUM = 16  # Effective batch: 16
LEARNING_RATE = 1e-4
EPOCHS = 3
WARMUP = 100
```

**Key script components:**
```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    "microsoft/phi-4",
    max_seq_length=4096,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)

# Train with reasoning-weighted loss
# Apply 2x weight to tokens inside <think> tags
```

### 4. Validation Framework
```
evaluation/
├── technical_accuracy.py   # Verify vs NotebookLM knowledge
├── llm_judge.py           # GPT-4/Claude reasoning quality
├── alignment_score.py     # NotebookLM citation accuracy
└── test_cases.jsonl       # 200+ held-out scenarios
```

**Metrics:**
```python
TARGETS = {
    "technical_accuracy": 0.95,      # vs NotebookLM ground truth
    "reasoning_quality": 4.5,        # LLM-as-Judge (1-5 scale)
    "notebooklm_alignment": 0.95,    # % claims with valid sources
    "config_syntax_valid": 1.0,      # 100% when configs included
    "compliance_accuracy": 1.0,      # Exact requirement citations
}
```

### 5. Deployment Package
```
deployment/
├── quantize_gguf.py       # CPU inference (Q4_K_M)
├── quantize_gptq.py       # GPU inference (4-bit)
├── inference.py           # Production inference
└── batch_process.py       # Bulk operations
```

**Inference script:**
```python
def infer(prompt: str) -> dict:
    """
    Returns:
        {
            "reasoning": "<think>...</think>",
            "answer": "Final design...",
            "sources": ["Book Ch3", "Guide §1.2"],
            "latency_ms": 2847
        }
    """
```

---

## IMPLEMENTATION SEQUENCE

### Phase 1: Knowledge Extraction (2 hours)
```bash
python knowledge_extraction/extract.py \
  --notebooklm-url "https://notebooklm.google.com/notebook/86b1a8b9-8a9c-486e-8fb3-2e2e0f914528" \
  --output-dir knowledge_extraction/

# Validates extraction completeness
python knowledge_extraction/validate.py
```

**Success criteria:**
- [ ] All 5 JSON files created
- [ ] Min 50 design patterns extracted
- [ ] Min 30 troubleshooting scenarios
- [ ] All compliance frameworks covered
- [ ] Min 100 config examples

### Phase 2: Dataset Generation (2 hours)
```bash
python data_generation/generate.py \
  --knowledge-dir knowledge_extraction/ \
  --output generated_data/training.jsonl \
  --count 10000 \
  --validate

# Split data
python data_generation/split.py \
  --input generated_data/training.jsonl \
  --train 0.9 --val 0.1
```

**Success criteria:**
- [ ] 10,000+ samples generated
- [ ] All samples pass quality gates
- [ ] 100% NotebookLM alignment
- [ ] No duplicate scenarios

### Phase 3: Fine-Tuning (12 hours)
```bash
python fine_tuning/train.py \
  --config fine_tuning/config.yaml \
  --train-data generated_data/train.jsonl \
  --val-data generated_data/val.jsonl \
  --output models/phi4-network-architect
```

**Success criteria:**
- [ ] Training completes 3 epochs
- [ ] Validation loss decreases
- [ ] No OOM errors
- [ ] Model saves successfully

### Phase 4: Validation (1 hour)
```bash
python evaluation/run_all.py \
  --model models/phi4-network-architect \
  --test-data evaluation/test_cases.jsonl \
  --knowledge-dir knowledge_extraction/

# Generates report
cat evaluation/results.json
```

**Success criteria:**
- [ ] Technical accuracy >95%
- [ ] Reasoning quality >4.5/5.0
- [ ] NotebookLM alignment >95%
- [ ] All configs syntactically valid

### Phase 5: Deployment (1 hour)
```bash
# Quantize for CPU
python deployment/quantize_gguf.py \
  --model models/phi4-network-architect \
  --output models/phi4-q4.gguf

# Quantize for GPU
python deployment/quantize_gptq.py \
  --model models/phi4-network-architect \
  --output models/phi4-gptq

# Test inference
python deployment/inference.py \
  --model models/phi4-gptq \
  --prompt "Design network for 1000 users, HIPAA compliance"
```

---

## PROJECT STRUCTURE

```
phi4-network-architect/
├── knowledge_extraction/
│   ├── extract.py              # Query NotebookLM → JSON
│   ├── validate.py             # Check extraction completeness
│   ├── design_patterns.json    # Output: topologies, criteria
│   ├── troubleshooting.json    # Output: diagnostic trees
│   ├── compliance.json         # Output: requirements
│   ├── vendor_configs.json     # Output: config patterns
│   └── cost_benchmarks.json    # Output: CapEx/OpEx/timelines
├── data_generation/
│   ├── generate.py             # Main generator
│   ├── reasoning_builder.py    # <think> chain creator
│   ├── validator.py            # Quality gates
│   ├── split.py               # Train/val split
│   └── config.yaml            # Generation parameters
├── fine_tuning/
│   ├── train.py               # Unsloth training script
│   ├── config.yaml            # Hyperparameters
│   └── requirements.txt
├── evaluation/
│   ├── run_all.py             # Execute all evals
│   ├── technical_accuracy.py  # vs NotebookLM
│   ├── llm_judge.py           # Reasoning quality
│   ├── alignment_score.py     # Citation accuracy
│   └── test_cases.jsonl       # 200+ scenarios
├── deployment/
│   ├── quantize_gguf.py       # CPU quantization
│   ├── quantize_gptq.py       # GPU quantization
│   ├── inference.py           # Production inference
│   └── batch_process.py       # Bulk processing
├── generated_data/
│   ├── training.jsonl         # Full dataset
│   ├── train.jsonl           # 90% training
│   └── val.jsonl             # 10% validation
├── models/
│   ├── phi4-network-architect/  # Fine-tuned model
│   ├── phi4-q4.gguf            # CPU quantized
│   └── phi4-gptq/              # GPU quantized
└── README.md
```

---

## CRITICAL REQUIREMENTS

### Knowledge Extraction
**Must extract from NotebookLM:**
1. Design patterns with decision criteria
2. Troubleshooting methodologies (step-by-step)
3. Compliance requirements (version-specific)
4. Vendor configuration patterns
5. Cost/sizing benchmarks

**Do NOT proceed to dataset generation without validated extraction.**

### Dataset Quality
**Every sample must:**
- Reference specific NotebookLM sources (min 5 per sample)
- Use `<think>` tags with 5-15 reasoning steps
- Apply book methodologies exactly as described
- Include version-specific compliance (e.g., PCI-DSS 4.0.1 §1.2.1)
- Use vendor-validated config syntax

**Reject samples that:**
- Lack source attribution
- Have <3 reasoning steps
- Contain generic/unsourced claims
- Use incorrect compliance versions
- Have invalid config syntax

### Training
**Phi-4 specific:**
- 4-bit quantization mandatory (VRAM constraint)
- LoRA rank 32 (14B model needs higher capacity)
- Batch size 1 + grad accum 16 (memory management)
- Learning rate 1e-4 (lower for larger model)
- Reasoning-weighted loss (2x weight on `<think>` tokens)

### Validation
**Must achieve:**
- Technical accuracy ≥95% vs NotebookLM
- Reasoning quality ≥4.5/5.0 (LLM-Judge)
- NotebookLM alignment ≥95%
- Config syntax validity 100%

**If targets not met:** Add more training data or adjust hyperparameters.

---

## QUALITY GATES

### Gate 1: Knowledge Extraction
- [ ] 50+ design patterns
- [ ] 30+ troubleshooting scenarios
- [ ] All compliance frameworks
- [ ] 100+ config examples
- [ ] Cost benchmarks present

### Gate 2: Dataset Generation
- [ ] 10,000+ samples
- [ ] 100% pass quality checks
- [ ] No duplicates
- [ ] Source distribution balanced

### Gate 3: Fine-Tuning
- [ ] Training converges
- [ ] Validation loss decreases
- [ ] No OOM errors
- [ ] Model saves cleanly

### Gate 4: Validation
- [ ] Technical accuracy ≥95%
- [ ] Reasoning quality ≥4.5/5.0
- [ ] Alignment ≥95%
- [ ] Syntax validity 100%

### Gate 5: Deployment
- [ ] Quantization successful
- [ ] Inference <3s
- [ ] Memory <10GB VRAM
- [ ] Output format correct

---

## DEPENDENCIES

```txt
# requirements.txt
torch>=2.0.0
transformers>=4.36.0
datasets>=2.14.0
unsloth>=2024.1
peft>=0.7.0
accelerate>=0.25.0
bitsandbytes>=0.41.3
sentencepiece>=0.1.99
tensorboard>=2.15.0
pyyaml>=6.0
tqdm>=4.66.0
```

---

## RESOURCE REQUIREMENTS

**Compute:**
- GPU: RTX 4090 (24GB) minimum
- CPU: 16+ cores recommended
- RAM: 64GB+ recommended

**Storage:**
- NotebookLM extraction: ~500MB
- Generated dataset: ~2GB
- Fine-tuned model: ~8GB (4-bit)
- Quantized models: ~5GB each

**Time:**
- Extraction: 2 hours
- Generation: 2 hours
- Training: 12 hours (overnight)
- Validation: 1 hour
- Deployment: 1 hour
- **Total: ~18 hours**

---

## SUCCESS METRICS

**Technical:**
- ✅ Accuracy: >95% vs NotebookLM
- ✅ Reasoning: >4.5/5.0 quality
- ✅ Alignment: >95% source accuracy
- ✅ Latency: <3s inference
- ✅ Memory: <10GB VRAM

**Qualitative:**
- ✅ Designs match book methodologies
- ✅ Every claim traces to source
- ✅ Configs are vendor-validated
- ✅ Compliance versions accurate
- ✅ Troubleshooting follows book methods

---

## ERROR HANDLING

**If extraction fails:**
- Check NotebookLM URL accessibility
- Verify query format
- Retry with simplified queries

**If generation fails quality:**
- Review extracted knowledge completeness
- Check reasoning_builder logic
- Validate source attribution

**If training OOMs:**
- Reduce batch size to 1
- Increase gradient accumulation
- Lower max_seq_length to 2048

**If validation fails:**
- Generate more diverse training data
- Adjust hyperparameters
- Retrain with higher quality samples

---

## EXAMPLE OUTPUT

**Input:**
```
Design network for 5000 users across 20 sites with PCI-DSS compliance and 99.99% uptime
```

**Output:**
```json
{
  "reasoning": "<think>
Step 1: Topology per [Enterprise Design Ch3 p87]
- 5000 users → three-tier required
- Collapsed core max 2500 per vendor guide
- Decision: Three-tier for scalability

Step 2: PCI-DSS [Compliance Guide 4.0.1 §1.2]
- CDE segmentation mandatory
- Dedicated firewall required
- Quarterly scans per §11.3

Step 3: HA [Reliability Ch8]
- 99.99% = 52.6min/year downtime
- N+1 core minimum
- Active-active DCs required

[...9 total steps...]
</think>",
  
  "answer": "Three-tier architecture with redundant core (Cisco Catalyst 9500), distribution (HSRP), access (802.1X). PCI-DSS CDE on isolated VLAN with dedicated Palo Alto firewalls. Active-active data centers with GSLB. Cost: $9.3M CapEx.",
  
  "sources": [
    "Enterprise Network Design Ch3 p87",
    "PCI-DSS 4.0.1 §1.2, §11.3",
    "Network Reliability Handbook Ch8"
  ],
  
  "latency_ms": 2847
}
```

---

## NOTES

- **NotebookLM URL:** https://notebooklm.google.com/notebook/86b1a8b9-8a9c-486e-8fb3-2e2e0f914528
- **Model:** microsoft/phi-4 (14B, reasoning-optimized)
- **Framework:** Unsloth (fast + memory efficient)
- **Quantization:** 4-bit for training, GGUF/GPTQ for deployment
- **Key innovation:** Every technical claim grounded in book sources

---

## BUILD IT

**Clone, install, execute:**
```bash
# Knowledge extraction
python knowledge_extraction/extract.py

# Dataset generation  
python data_generation/generate.py --count 10000

# Fine-tune (overnight)
python fine_tuning/train.py

# Validate
python evaluation/run_all.py

# Deploy
python deployment/quantize_gptq.py && python deployment/inference.py
```

**Timeline:** 18 hours. Most is unattended.

**Output:** Production Phi-4 network architect with book-validated knowledge.

**Ship it.**
