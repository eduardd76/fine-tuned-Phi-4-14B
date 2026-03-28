# Claude Code Prompt: Fine-Tuning Phi-4 for Virtual Network Architecture (NotebookLM Enhanced)

## Critical Context

I have a **NotebookLM instance** https://notebooklm.google.com/notebook/86b1a8b9-8a9c-486e-8fb3-2e2e0f914528 containing comprehensive network design books, vendor guides, and technical documentation. You MUST use this knowledge base extensively to create technically accurate, industry-standard training data.

**NotebookLM contains:**
- Network architecture design books (CCDE-level content)
- Vendor design guides (Cisco, Juniper, Arista, etc.)
- Troubleshooting methodologies
- Compliance implementation guides (PCI-DSS, HIPAA, SOX, etc.)
- Modern technology guides (SD-WAN, SASE, Zero Trust, VXLAN/EVPN)
- Real-world case studies and design patterns

## Objective

Build a complete fine-tuning pipeline to create a reasoning model based on **Phi-4 (14B parameters)** that acts as a virtual network architect. The model must use chain-of-thought (CoT) reasoning with `<think>` tags, incorporating actual industry methodologies and design patterns from the NotebookLM knowledge base.

## Base Model Specification

**Model:** `microsoft/phi-4` (14B parameters)
**Why Phi-4:**
- Superior reasoning capabilities (4.6/5.0 on reasoning benchmarks)
- Excellent multi-step problem solving
- Strong technical domain performance
- Multi-modal capabilities (can reason about network diagrams)
- Microsoft research quality
- 8.5GB VRAM with 4-bit quantization

**HuggingFace:** https://huggingface.co/microsoft/phi-4

## Critical Requirement: NotebookLM Integration

### Phase 1: Extract Knowledge from NotebookLM

Before generating any training data, you MUST:

1. **Query NotebookLM for Design Methodologies**
   - "What are the standard network topology design patterns?"
   - "How do you size networks for different user counts?"
   - "What are the decision criteria for collapsed core vs three-tier?"
   - "When should you use BGP vs OSPF vs EIGRP?"
   - "What are the SD-WAN design best practices?"
   - "How do you design for high availability?"
   - Extract ALL methodology sections

2. **Query NotebookLM for Troubleshooting Approaches**
   - "What is the systematic troubleshooting methodology for BGP?"
   - "How do you diagnose packet loss?"
   - "What are common routing loop causes and fixes?"
   - "How do you troubleshoot QoS issues?"
   - "What is the Layer 1-7 diagnostic sequence?"
   - Extract troubleshooting decision trees

3. **Query NotebookLM for Technical Specifications**
   - "What are PCI-DSS network segmentation requirements?"
   - "What are HIPAA encryption requirements?"
   - "What are the Cisco VSS configuration requirements?"
   - "What are VXLAN/EVPN design considerations?"
   - "What are SD-WAN dual-transport best practices?"
   - Extract specific technical requirements

4. **Query NotebookLM for Real-World Constraints**
   - "What are typical CapEx costs for enterprise networks?"
   - "What are staffing requirements by network size?"
   - "What are vendor selection criteria?"
   - "What are common implementation timelines?"
   - Extract realistic constraints and metrics

5. **Query NotebookLM for Configuration Examples**
   - "What are BGP configuration best practices?"
   - "What are proper ACL structures for compliance?"
   - "What are HSRP/VRRP configuration patterns?"
   - "What are QoS policy examples?"
   - Extract actual configuration templates

### Phase 2: Create Knowledge Extraction Module

Build a Python module that:

```python
# knowledge_extractor.py

class NotebookLMKnowledgeExtractor:
    """
    Extracts and structures knowledge from NotebookLM for dataset generation.
    
    This module queries NotebookLM and creates structured knowledge bases for:
    - Design methodologies
    - Troubleshooting procedures
    - Technical specifications
    - Configuration templates
    - Real-world constraints
    """
    
    def extract_design_patterns(self) -> Dict[str, Any]:
        """
        Query NotebookLM for design patterns and return structured data.
        
        Returns:
            {
                "topology_patterns": {
                    "collapsed_core": {
                        "user_range": "50-2000",
                        "pros": [...],
                        "cons": [...],
                        "use_cases": [...],
                        "source_book": "..."
                    },
                    "three_tier": {...},
                    "spine_leaf": {...}
                },
                "wan_patterns": {...},
                "security_patterns": {...}
            }
        """
        pass
    
    def extract_troubleshooting_methodologies(self) -> Dict[str, Any]:
        """
        Query NotebookLM for troubleshooting approaches.
        
        Returns structured diagnostic decision trees for:
        - BGP issues
        - OSPF issues
        - Packet loss
        - Performance problems
        - Security incidents
        """
        pass
    
    def extract_compliance_requirements(self) -> Dict[str, Any]:
        """
        Query NotebookLM for compliance specifications.
        
        Returns exact requirements for PCI-DSS, HIPAA, SOX, etc.
        """
        pass
    
    def extract_vendor_specifics(self) -> Dict[str, Any]:
        """
        Query NotebookLM for vendor-specific details.
        
        Returns configuration syntax, best practices, limitations
        for Cisco, Juniper, Arista, etc.
        """
        pass
```

### Phase 3: Dataset Generation Requirements

Generate **10,000+ training samples** where EVERY sample is grounded in NotebookLM knowledge.

#### Dataset Structure with NotebookLM Integration

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Design a network for an enterprise financial services company with 8,000 users across 25 sites. We must comply with PCI-DSS and SOX, need 99.99% uptime, and have a flexible budget."
    },
    {
      "role": "assistant",
      "content": "<think>
Step 1: Reference design methodology from NotebookLM
- Source: [Book name] Chapter X - Enterprise Network Sizing
- 8,000 users falls into 'large enterprise' category
- Recommended: Three-tier architecture per industry standard
- Justification: Supports growth to 50k users, provides fault isolation

Step 2: Analyze PCI-DSS requirements (from NotebookLM compliance guide)
- PCI-DSS 4.0.1 Section 1.2: Network segmentation required
- Must create isolated Cardholder Data Environment (CDE)
- CDE requires: Dedicated firewalls, no direct internet access, quarterly scans
- Specific requirement: Section 1.3 - 'prohibit direct public access between internet and CDE'

Step 3: Calculate uptime requirements
- 99.99% = 52.6 minutes downtime/year
- Reference: [Book name] HA design patterns
- Requires: N+1 redundancy minimum, preferably N+2
- Active-active data centers with sub-second failover
- No single point of failure across entire stack

Step 4: Apply three-tier topology decision tree (from NotebookLM)
- User count: 8,000 ✓ (exceeds 5,000 threshold)
- Site count: 25 ✓ (exceeds 10 threshold)
- Budget: Flexible ✓
- Compliance: Critical ✓
- Decision matrix score: 95/100 for three-tier
- Alternatives considered:
  * Collapsed core: Rejected - scale too large (max 5,000 users per best practice)
  * Spine-leaf: Rejected - data center topology, not campus/enterprise

Step 5: WAN design based on NotebookLM SD-WAN guide
- Reference: [Book name] Chapter Y - SD-WAN for Financial Services
- 25 sites requires centralized management
- Financial services needs: Predictable latency for trading apps
- Dual transport recommended: MPLS (primary) + Internet (secondary)
- Per-site bandwidth calculation: 8000 users / 25 sites ≈ 320 users/site
- Formula from book: 320 users × 2 Mbps average = 640 Mbps minimum
- Recommended: 1 Gbps per site with QoS (20% headroom)

Step 6: Security architecture from Zero Trust framework (NotebookLM)
- PCI-DSS requires micro-segmentation
- Reference: [Security book] Zero Trust implementation patterns
- Layer 1: Network segmentation (VLANs + firewalls)
- Layer 2: Identity-based access (802.1X + NAC)
- Layer 3: Continuous authentication
- Layer 4: Encrypted communications (TLS 1.3 minimum per PCI-DSS 4.0)

Step 7: Technology stack selection (from NotebookLM vendor comparison)
- Core switches: Cisco Catalyst 9500 vs Arista 7500R
  * Decision: Cisco - better financial services market penetration
  * PCI validation: Cisco has PCI-certified designs
- NGFW: Palo Alto PA-5450 
  * Justification: High throughput (40 Gbps with IPS)
  * PCI requirement: Application-aware firewalling ✓
- Load balancing: F5 BIG-IP
  * Financial services standard per NotebookLM case studies

Step 8: Cost estimation using NotebookLM benchmarks
- Reference: [Book name] Chapter Z - Enterprise TCO models
- Hardware multiplier: 8,000 users × $1,000/user = $8M baseline
- Compliance premium: 30% for PCI-DSS (additional security)
- HA premium: 25% for 99.99% uptime (full redundancy)
- Total CapEx estimate: $8M × 1.55 = $12.4M

Step 9: Implementation phasing from NotebookLM project management guide
- Reference: [Book name] Enterprise Migration Patterns
- Phase 1 (Months 1-2): Core infrastructure (critical path)
- Phase 2 (Months 3-4): Site rollout (parallel deployment)
- Phase 3 (Months 5-6): PCI audit prep and validation
- Total timeline: 6 months (standard for this scale)
</think>

## Network Architecture Design

**Core Topology:** Three-tier architecture (Core-Distribution-Access) based on enterprise design patterns for 8,000+ user deployments.

**Rationale:** Industry-standard approach from [Book name] for large enterprises. Provides horizontal scalability to 50,000 users, fault isolation between layers, and simplified troubleshooting per CCDE design principles.

**WAN Connectivity:** SD-WAN with dual transport:
- Primary: MPLS (guaranteed latency for trading applications)
- Secondary: Internet (cost-effective bandwidth for non-critical traffic)
- Per-site bandwidth: 1 Gbps (calculated: 320 users × 2 Mbps + 20% headroom)
- QoS policy: EF class for voice, AF41 for trading apps, best-effort for general traffic

**PCI-DSS Compliance Architecture:**
- **CDE Segmentation:** Isolated VLAN 100 with dedicated Palo Alto PA-5450 firewalls
- **Access Control:** 802.1X with Cisco ISE, multi-factor authentication mandatory
- **Encryption:** TLS 1.3 in transit (PCI-DSS 4.0.1 requirement), AES-256-GCM at rest
- **Monitoring:** Quarterly vulnerability scans (PCI requirement), annual penetration testing
- **Audit:** All CDE access logged to Splunk SIEM with 7-year retention (SOX requirement)

**High Availability Design:**
To achieve 99.99% uptime (52.6 minutes max downtime/year):
- Active-active data centers with GSLB (Global Server Load Balancing)
- N+1 redundancy at core layer (3 switches, 2 required)
- Dual-attached distribution switches with HSRP
- Automated failover with BFD (sub-second detection)
- Separate physical paths for primary/backup WAN links

**Technology Stack:**
- **Core:** Cisco Catalyst 9500-48Y4C (3× for N+1 redundancy)
- **Distribution:** Cisco Catalyst 9400 with redundant supervisors
- **NGFW:** Palo Alto PA-5450 (40 Gbps throughput with IPS enabled)
- **Load Balancing:** F5 BIG-IP i5800 (industry standard for financial services)
- **SD-WAN:** Cisco Viptela (tight integration with Catalyst platform)

**Cost Estimation:**
- CapEx: $12.4M (hardware $5.6M, software $3.7M, services $2.4M, contingency $0.7M)
- Annual OpEx: $2.1M (WAN circuits $0.8M, maintenance $0.7M, cloud $0.4M, personnel $0.2M)
- 3-Year TCO: $18.7M

**Implementation Roadmap:**
- Phase 1 (Months 1-2): Deploy core/distribution at primary and DR data centers
- Phase 2 (Months 3-4): Site rollout in waves of 5 sites (parallel deployment)
- Phase 3 (Months 5-6): PCI-DSS validation, SOX audit preparation, user migration

**References:**
- Topology design: [Book name] Chapter X
- PCI-DSS requirements: [Compliance guide] Section Y
- HA patterns: [Book name] Chapter Z
- TCO model: [Book name] Enterprise cost analysis"
    }
  ],
  "metadata": {
    "notebooklm_sources": [
      "[Book name] - Enterprise Network Design",
      "[Compliance guide] - PCI-DSS Implementation",
      "[Vendor guide] - Cisco Enterprise Architecture"
    ],
    "design_patterns_used": [
      "Three-tier topology",
      "SD-WAN dual transport",
      "Zero Trust segmentation"
    ],
    "compliance_requirements": ["PCI-DSS 4.0.1", "SOX"],
    "reasoning_steps": 9
  }
}
```

## Required Deliverables

### 1. NotebookLM Knowledge Extraction Module

Create `knowledge_extraction/` directory with:

```
knowledge_extraction/
├── extract_from_notebooklm.py       # Main extraction script
├── design_patterns.json             # Extracted design methodologies
├── troubleshooting_trees.json       # Diagnostic decision trees
├── compliance_requirements.json     # Specific regulatory details
├── vendor_specifics.json            # Cisco/Juniper/Arista details
├── configuration_templates.json     # Real config examples
├── cost_benchmarks.json             # CapEx/OpEx data from books
└── README.md                        # How to query NotebookLM
```

**Extraction Script Must:**
1. Query NotebookLM systematically for all knowledge areas
2. Structure extracted knowledge into JSON format
3. Create indexes for fast lookup during dataset generation
4. Validate extracted knowledge completeness
5. Generate knowledge coverage report

### 2. Enhanced Dataset Generator

The dataset generator MUST:

1. **Load NotebookLM knowledge** before generating any samples
2. **Reference specific books/chapters** in reasoning chains
3. **Use exact methodologies** from extracted knowledge
4. **Apply actual decision trees** from troubleshooting guides
5. **Include specific compliance requirements** with version numbers
6. **Use vendor-specific configurations** from guides
7. **Apply realistic cost models** from extracted benchmarks

#### Dataset Characteristics:

**Scale:** 10,000+ samples
- 7,000 design scenarios (70%)
- 3,000 troubleshooting scenarios (30%)

**Quality Requirements:**
- ✅ Every reasoning chain references NotebookLM sources
- ✅ Every technical decision uses book methodology
- ✅ Every configuration example is vendor-validated
- ✅ Every compliance requirement includes specific version
- ✅ Every cost estimate uses benchmark data from books
- ✅ Every troubleshooting follows extracted decision tree

**Reasoning Chain Structure:**
```
<think>
Step 1: [Analysis] - Reference: [Book] Chapter X
Step 2: [Constraint evaluation] - Per [Guide] Section Y
Step 3: [Alternative comparison] - Using [Book] decision matrix
Step 4: [Decision] - Based on [Methodology] from [Source]
Step 5: [Implementation planning] - Following [Book] best practices
[... 5-15 steps total ...]
</think>
```

### 3. Validation Framework

Create validation module that ensures:

```python
# validation/notebooklm_validator.py

class NotebookLMValidator:
    """Validates that generated samples correctly use NotebookLM knowledge"""
    
    def validate_reasoning_chain(self, sample: Dict) -> ValidationResult:
        """
        Checks:
        - Does reasoning reference specific NotebookLM sources?
        - Are technical details consistent with extracted knowledge?
        - Are methodologies applied correctly?
        - Are configurations syntactically valid per vendor guides?
        """
        pass
    
    def validate_technical_accuracy(self, sample: Dict) -> ValidationResult:
        """
        Verifies against NotebookLM knowledge base:
        - Protocol usage correctness
        - IP addressing validity
        - Compliance requirement completeness
        - Cost estimate realism
        """
        pass
    
    def validate_source_attribution(self, sample: Dict) -> ValidationResult:
        """
        Ensures every technical claim can be traced to NotebookLM source
        """
        pass
```

### 4. Fine-Tuning Script for Phi-4

```python
from unsloth import FastLanguageModel
import torch

# Load Phi-4
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="microsoft/phi-4",
    max_seq_length=4096,  # Phi-4 supports up to 16K
    dtype=None,
    load_in_4bit=True,
)

# LoRA configuration optimized for Phi-4 reasoning
model = FastLanguageModel.get_peft_model(
    model,
    r=32,  # Higher rank for 14B model complex reasoning
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

# Training arguments optimized for Phi-4
training_args = TrainingArguments(
    output_dir="./phi4-network-architect",
    num_train_epochs=3,
    per_device_train_batch_size=1,  # Phi-4 is larger, reduce batch size
    gradient_accumulation_steps=16,  # Effective batch size: 16
    learning_rate=1e-4,  # Slightly lower for larger model
    warmup_steps=100,
    logging_steps=10,
    save_steps=500,
    bf16=True,
    optim="adamw_8bit",
    max_grad_norm=1.0,
    weight_decay=0.01,
)
```

### 5. Evaluation Framework

#### A. Technical Accuracy Benchmarks
```python
# evaluation/technical_accuracy.py

class TechnicalAccuracyEvaluator:
    """
    Evaluates generated designs against NotebookLM knowledge base
    """
    
    def evaluate_topology_selection(self, prediction: str, context: Dict) -> float:
        """Score topology choice against book decision criteria"""
        pass
    
    def evaluate_protocol_usage(self, prediction: str) -> float:
        """Verify BGP/OSPF/EIGRP usage matches extracted guidelines"""
        pass
    
    def evaluate_compliance(self, prediction: str, requirements: List[str]) -> float:
        """Check if all compliance requirements from books are met"""
        pass
    
    def evaluate_configuration_syntax(self, config: str, vendor: str) -> float:
        """Validate config syntax against vendor guide examples"""
        pass
```

#### B. Reasoning Quality (LLM-as-Judge)
```python
# evaluation/llm_judge.py

def evaluate_reasoning_quality(sample: Dict, notebooklm_knowledge: Dict) -> Dict[str, float]:
    """
    Use Claude/GPT-4 to judge:
    - Logical coherence (1-5)
    - Methodology adherence to NotebookLM sources (1-5)
    - Completeness of analysis (1-5)
    - Source attribution accuracy (1-5)
    - Practical applicability (1-5)
    """
    pass
```

#### C. NotebookLM Alignment Score
```python
# evaluation/notebooklm_alignment.py

def calculate_alignment_score(sample: Dict, knowledge_base: Dict) -> float:
    """
    Measures how well generated sample aligns with NotebookLM knowledge:
    - Percentage of claims that reference valid sources
    - Accuracy of book/chapter citations
    - Correctness of applied methodologies
    - Consistency with extracted patterns
    
    Target: >95% alignment for production use
    """
    pass
```

### 6. Complete Project Structure

```
phi4-network-architect-notebooklm/
├── knowledge_extraction/
│   ├── extract_from_notebooklm.py       # Query NotebookLM systematically
│   ├── design_patterns.json             # Extracted topologies, methodologies
│   ├── troubleshooting_trees.json       # Diagnostic decision trees
│   ├── compliance_requirements.json     # PCI-DSS, HIPAA, SOX specifics
│   ├── vendor_specifics.json            # Cisco, Juniper, Arista details
│   ├── configuration_templates.json     # Real config examples from books
│   ├── cost_benchmarks.json             # CapEx/OpEx from case studies
│   └── knowledge_coverage_report.md     # What was extracted
├── data_generation/
│   ├── dataset_generator.py             # Main generator using NotebookLM knowledge
│   ├── reasoning_chain_builder.py       # Builds <think> chains from book methodologies
│   ├── technical_details_injector.py    # Adds vendor-specific accuracy
│   ├── source_attribution.py            # Ensures book references in output
│   └── diversity_analyzer.py            # Ensures coverage of all extracted patterns
├── validation/
│   ├── notebooklm_validator.py          # Validates against knowledge base
│   ├── technical_accuracy_checker.py    # Verifies protocols, configs, compliance
│   ├── source_verification.py           # Checks book references are valid
│   └── reasoning_quality_checker.py     # Validates <think> chain logic
├── fine_tuning/
│   ├── train_phi4.py                    # Phi-4 specific training script
│   ├── config.yaml                      # Hyperparameters optimized for Phi-4
│   ├── custom_loss.py                   # Loss function that weights reasoning
│   └── requirements.txt
├── evaluation/
│   ├── benchmark_suite.py               # Technical accuracy tests
│   ├── llm_judge.py                     # GPT-4/Claude reasoning evaluation
│   ├── notebooklm_alignment.py          # Measures knowledge base fidelity
│   ├── comparison_baseline.py           # Compare vs base Phi-4
│   └── test_cases.jsonl                 # 200+ test scenarios from books
├── deployment/
│   ├── quantize_gguf.py                 # CPU deployment
│   ├── quantize_gptq.py                 # GPU deployment  
│   ├── inference.py                     # Production inference with CoT
│   └── batch_inference.py
├── generated_data/
│   ├── training_data.jsonl              # 10k+ samples with NotebookLM grounding
│   ├── validation_data.jsonl            # 1k samples
│   └── test_data.jsonl                  # 500 samples
├── notebooks/
│   └── explore_notebooklm_knowledge.ipynb  # Interactive exploration
├── docs/
│   ├── NOTEBOOKLM_INTEGRATION.md        # How NotebookLM is used
│   ├── KNOWLEDGE_EXTRACTION_GUIDE.md    # Querying strategies
│   ├── DATASET_QUALITY_REPORT.md        # Analysis of generated data
│   └── SOURCE_ATTRIBUTION_INDEX.md      # Map samples → book sources
├── README.md
├── QUICKSTART.md
└── requirements.txt
```

## Specific Implementation Requirements

### Knowledge Extraction Phase (Critical!)

**Before generating ANY training data, you must:**

1. **Query NotebookLM for ALL design patterns:**
   ```
   Queries:
   - "What are all network topology types and their selection criteria?"
   - "What are the user count ranges for each topology?"
   - "What are the pros and cons of each design pattern?"
   - "What are real-world case studies for each pattern?"
   ```

2. **Extract complete troubleshooting methodologies:**
   ```
   Queries:
   - "What is the complete troubleshooting methodology for BGP?"
   - "What are the diagnostic steps for packet loss?"
   - "What are Layer 1-7 troubleshooting sequences?"
   - "What are common issues and their symptoms?"
   ```

3. **Get exact compliance specifications:**
   ```
   Queries:
   - "What are the exact PCI-DSS 4.0.1 network requirements?"
   - "What are HIPAA encryption requirements with versions?"
   - "What are SOX audit logging requirements?"
   - "What are specific technical controls for each regulation?"
   ```

4. **Extract vendor configuration patterns:**
   ```
   Queries:
   - "What are Cisco BGP configuration best practices?"
   - "What are Juniper OSPF configuration examples?"
   - "What are proper ACL structures from vendor guides?"
   - "What are QoS policy templates?"
   ```

5. **Get realistic cost and sizing data:**
   ```
   Queries:
   - "What are typical CapEx costs by user count?"
   - "What are OpEx percentages for different components?"
   - "What are staffing requirements by network size?"
   - "What are implementation timelines for different scales?"
   ```

### Dataset Generation Requirements

**Every generated sample MUST:**

1. ✅ Reference specific NotebookLM sources in `<think>` tags
2. ✅ Apply methodologies exactly as described in books
3. ✅ Use decision trees from troubleshooting guides
4. ✅ Include version-specific compliance requirements
5. ✅ Use vendor-validated configuration syntax
6. ✅ Apply realistic cost models from extracted benchmarks
7. ✅ Follow implementation patterns from case studies

**Sample Quality Gates:**
- Minimum 5 references to NotebookLM sources per sample
- 100% technical accuracy against knowledge base
- 100% valid configuration syntax when included
- 100% correct compliance requirement citations
- 95%+ alignment score with extracted methodologies

### Training Requirements

**Phi-4 Specific Optimizations:**

```python
# Fine-tuning configuration for Phi-4's reasoning capabilities

training_config = {
    "model": "microsoft/phi-4",
    "max_seq_length": 4096,  # Phi-4 handles longer context well
    "lora_rank": 32,  # Higher for 14B model
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "learning_rate": 1e-4,  # Lower for larger model stability
    "batch_size": 1,  # Phi-4 needs more memory
    "gradient_accumulation": 16,  # Effective batch: 16
    "epochs": 3,
    "warmup_steps": 100,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
}

# Custom loss that weights reasoning tokens higher
def reasoning_aware_loss(logits, labels, reasoning_mask):
    """
    Apply 2x weight to tokens inside <think> tags
    This encourages higher quality reasoning chains
    """
    pass
```

### Evaluation Requirements

**Must compare:**
1. Base Phi-4 (zero-shot) vs Fine-tuned Phi-4
2. Fine-tuned Phi-4 vs Human expert (on subset)
3. Different training data quantities (2k, 5k, 10k samples)

**Metrics to report:**
- Technical accuracy: Target >95% (vs NotebookLM ground truth)
- Reasoning quality: Target >4.5/5.0 (LLM-as-Judge)
- NotebookLM alignment: Target >95%
- Source attribution accuracy: Target 100%
- Configuration syntax validity: Target 100%

## Success Criteria

The fine-tuned Phi-4 model should:

1. ✅ Design networks using methodologies from NotebookLM books
2. ✅ Reference specific book chapters/sections in reasoning
3. ✅ Apply troubleshooting decision trees from guides
4. ✅ Include exact compliance requirements with versions
5. ✅ Generate vendor-validated configurations
6. ✅ Use realistic cost estimates from benchmarks
7. ✅ Show reasoning chains matching expert architect thinking
8. ✅ Achieve >95% technical accuracy against book knowledge
9. ✅ Produce auditable, verifiable, source-grounded outputs
10. ✅ Generalize book knowledge to novel scenarios

## Performance Targets

- **Training**: <12 hours on RTX 4090 (10k samples, Phi-4 14B)
- **Inference**: <3 seconds per response (including reasoning chain)
- **Memory**: Fit in 24GB VRAM with 4-bit quantization
- **Accuracy**: >95% technical accuracy (validated against NotebookLM)
- **Reasoning**: >4.5/5.0 quality (LLM-as-Judge)
- **Alignment**: >95% with NotebookLM knowledge base

## Documentation Requirements

### 1. NOTEBOOKLM_INTEGRATION.md
Document:
- How to query NotebookLM effectively
- What knowledge was extracted
- How extracted knowledge is used in dataset generation
- Quality assurance processes

### 2. KNOWLEDGE_COVERAGE_REPORT.md
Report:
- Complete list of extracted design patterns
- Coverage of troubleshooting scenarios
- Compliance requirements extracted
- Vendor-specific details captured
- Gaps in knowledge base

### 3. SOURCE_ATTRIBUTION_INDEX.md
Create:
- Map from training samples → NotebookLM sources
- Verification that all claims are grounded
- Quality metrics for source usage

### 4. DATASET_QUALITY_REPORT.md
Analyze:
- Distribution of design patterns used
- Coverage of troubleshooting types
- Compliance requirement representation
- Vendor diversity
- Source attribution statistics

## Code Quality Requirements

1. **Type hints** for all functions
2. **Comprehensive docstrings** with NotebookLM source references
3. **Unit tests** for knowledge extraction and validation
4. **Integration tests** for end-to-end pipeline
5. **Error handling** with detailed logging
6. **Progress tracking** with rich progress bars
7. **Configuration** via YAML files
8. **Reproducibility** with random seeds

## Expected Deliverables

After completion, I should have:

1. ✅ **Knowledge Base**: Structured JSON from NotebookLM extraction
2. ✅ **Dataset**: 10,000+ samples grounded in book knowledge
3. ✅ **Fine-tuned Model**: Phi-4 specialized for network architecture
4. ✅ **Validation Report**: Technical accuracy >95% vs books
5. ✅ **Alignment Report**: >95% consistency with NotebookLM
6. ✅ **Deployment Package**: Quantized models for CPU/GPU
7. ✅ **Documentation**: Complete guides with source attribution

## Critical Success Factors

1. **Knowledge Extraction Quality**: Must capture ALL relevant knowledge from NotebookLM
2. **Source Grounding**: Every technical claim must trace to book source
3. **Methodology Fidelity**: Must apply book methodologies exactly as described
4. **Technical Accuracy**: Configurations must match vendor guide examples
5. **Reasoning Transparency**: Every decision must show book-based reasoning

## Notes for Claude Code

- **PRIORITY 1**: Extract knowledge from NotebookLM BEFORE generating data
- **PRIORITY 2**: Validate every sample against extracted knowledge
- **PRIORITY 3**: Ensure source attribution in all reasoning chains

- Use modern Python 3.10+ with type hints
- Optimize for Phi-4's 14B parameter size (requires more VRAM than 7B)
- Make extensive use of NotebookLM queries - don't rely on general knowledge
- Create comprehensive validation to ensure book fidelity
- Document which book informed each aspect of generated samples
- Include progress tracking and detailed logging
- Make system extensible for future NotebookLM updates

## Final Note

The key differentiator of this approach is **knowledge grounding**. Every technical decision, methodology, and recommendation in the training data must be traceable to specific books in NotebookLM. This creates a model that doesn't just generate plausible-sounding advice, but provides expert-level guidance grounded in industry-proven methodologies.

**Generate the complete implementation with NotebookLM integration as the foundation.**
