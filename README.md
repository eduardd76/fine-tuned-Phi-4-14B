# Phi-4 Network Architect

Fine-tuned **Microsoft Phi-4-14B** for CCDE-level network design, troubleshooting, and cost estimation. Produces chain-of-thought reasoning inside `<think>` blocks before answering.

Part of the [Dream Team agentic-netops-mvp](https://github.com/eduardd76/agentic-netops-mvp) project — this model serves as the **Virtual Architect** specialist agent.

---

## Quick Start (5 minutes to launch on AWS)

```bash
# 1. Clone the repo
git clone https://github.com/eduardd76/fine-tuned-Phi-4-14B.git
cd fine-tuned-Phi-4-14B

# 2. Set your AWS region and launch a GPU instance
export AWS_DEFAULT_REGION=us-east-1
bash aws_deployment/launch_ec2.sh

# 3. SSH in (credentials written to connection_details.txt)
ssh -i ~/.ssh/phi4-key.pem ubuntu@<INSTANCE_IP>

# 4. Run the full pipeline (downloads data, trains, quantizes, starts API)
bash run_full_pipeline.sh
```

The API will be live at `http://<INSTANCE_IP>:8000/api/v1/health` when Stage 8 completes (~14 hours total).

---

## Detailed Setup Guide

### Prerequisites

| Tool | Minimum version | Purpose |
|------|----------------|---------|
| AWS CLI | 2.x | EC2/S3 provisioning |
| Terraform | 1.5+ | IaC (optional, alternative to bash scripts) |
| Docker | 24+ | Local containerised serving |
| Python | 3.11 | Local development |
| CUDA | 12.1 | GPU training/inference |

### Option A — AWS EC2 (Recommended for training)

```bash
# Launch g4dn.xlarge (T4 16 GB) — cheapest option
bash aws_deployment/launch_ec2.sh

# Or use a spot instance (70 % cheaper, may be interrupted)
USE_SPOT=true bash aws_deployment/launch_ec2.sh

# Or use Terraform
cd aws_deployment/terraform
terraform init
terraform apply -var="ssh_public_key=$(cat ~/.ssh/id_rsa.pub)"
```

Once on the instance, run the automated setup (idempotent — safe to re-run):

```bash
bash setup_instance.sh
```

This installs: CUDA toolkit, Python 3.11 venv, PyTorch, Unsloth, TRL, PEFT, bitsandbytes, vLLM, FastAPI, AWS CloudWatch agent.

### Option B — Docker (Recommended for serving)

```bash
# Start the full stack (API + MCP server + Nginx)
cd deployment_artifacts/docker
docker-compose up -d

# Check health
curl http://localhost/health
```

GPU allocation is handled automatically via NVIDIA Container Toolkit. The Nginx proxy applies rate limiting (10 req/min per IP) and security headers.

### Option C — Local development

```bash
python3.11 -m venv ~/phi4-env
source ~/phi4-env/bin/activate
pip install -r fine_tuning/requirements.txt

# Start API with a pre-trained model
MODEL_PATH=/path/to/model uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## Pipeline Stages

`run_full_pipeline.sh` executes 8 stages with automatic checkpointing. Re-running after an interruption skips completed stages.

| Stage | Description | Duration |
|-------|-------------|----------|
| 1 | Knowledge validation (NotebookLM MCP) | 2 min |
| 2 | Dataset generation (10 k CCDE samples) | 20 min |
| 3 | Train/val/test split with stratification | 1 min |
| 4 | Unsloth LoRA fine-tuning (Phi-4-14B) | ~12 h |
| 5 | Evaluation against CCDE test cases | 30 min |
| 6 | GPTQ (GPU) + GGUF Q4\_K\_M (CPU) quantization | 45 min |
| 7 | S3 backup + EBS snapshot | 10 min |
| 8 | Dream Team integration setup + API start | 5 min |

```bash
# Resume after interruption
bash run_full_pipeline.sh --resume

# Run a single stage
bash run_full_pipeline.sh --stage 4
```

### Spot instance interruption handling

AWS sends SIGTERM 2 minutes before reclaiming a spot instance. The pipeline traps this signal, checkpoints the current stage, and syncs artefacts to S3 automatically.

### CUDA OOM recovery

If GPU runs out of memory during training, the pipeline automatically doubles `--gradient_accumulation_steps` and retries (up to 3 times) while keeping the effective batch size constant.

---

## Configuration Options

All configuration is via environment variables. Copy `.env.example` to `.env` and edit.

### Core settings

```bash
# Model
BASE_MODEL=microsoft/phi-4          # HuggingFace base model
MODEL_PATH=/data/models/phi4-network-architect

# Training
BATCH_SIZE=4                        # Per-GPU batch size
GRAD_ACCUM=16                       # Effective batch = 4 * 16 = 64
EPOCHS=3
LORA_RANK=32
LORA_ALPHA=32
MAX_SEQ_LEN=4096

# API
PHI4_API_KEY=change-me              # X-API-Key header value
RATE_LIMIT_RPM=30                   # Requests per minute per IP

# AWS
AWS_DEFAULT_REGION=us-east-1
S3_BUCKET=phi4-artefacts-<account>
AUTO_SHUTDOWN=true                  # Shutdown instance after pipeline completes
```

### Cost optimisation

```bash
# Spot instances (70 % savings, interruption risk)
USE_SPOT=true bash aws_deployment/launch_ec2.sh

# Auto-shutdown when training finishes
AUTO_SHUTDOWN=true bash run_full_pipeline.sh

# Pause Hugging Face endpoints when idle
# (endpoints charge ~$0.60/hr even when unused)
```

---

## API Documentation

Base URL: `http://<host>:8000/api/v1`

Authentication: `X-API-Key: <PHI4_API_KEY>` header required on all endpoints except `/health`.

Interactive docs: `http://<host>:8000/docs`

### POST /design

Generate a network architecture design.

**Request**

```json
{
  "users": 5000,
  "sites": 20,
  "uptime": 99.99,
  "compliance": ["PCI-DSS"],
  "data_center": false,
  "wireless": true,
  "wan_type": "SD-WAN",
  "budget": 2000000
}
```

**Response**

```json
{
  "reasoning": "Step 1: Analyse scale...\nStep 2: Select topology...",
  "design": "Three-tier campus with SD-WAN overlay...",
  "cost_estimate": {
    "capex": 1200000,
    "annual_opex": 180000,
    "implementation": 120000,
    "currency": "USD",
    "notes": "Includes 3-year hardware refresh cycle"
  },
  "sources": ["CCNP Enterprise Design ENSLD", "RFC 4271", "PCI-DSS 4.0.1 Req 1.3"],
  "confidence": 0.88,
  "latency_ms": 4230,
  "requires_human_review": true
}
```

`requires_human_review` is `true` when confidence < 0.80, compliance is specified, or uptime ≥ 99.99 %.

### POST /troubleshoot

Diagnose a network fault.

**Request**

```json
{
  "symptom": "BGP neighbor 10.0.0.1 flapping every 90 seconds",
  "device": "core-rtr-01",
  "protocol": "BGP",
  "additional_context": "show bgp summary output attached"
}
```

**Response**

```json
{
  "reasoning": "Step 1: Check BGP timers...",
  "diagnosis": "MAXPREFIX threshold exceeded causing BGP session reset",
  "remediation_steps": [
    "show bgp neighbors 10.0.0.1 | inc Maximum",
    "neighbor 10.0.0.1 maximum-prefix 10000 90 restart 5"
  ],
  "sources": ["RFC 4271 Section 8.2.2"],
  "confidence": 0.91,
  "severity": "high"
}
```

### POST /estimate

Generate a cost estimate.

**Request**

```json
{
  "users": 1000,
  "sites": 5,
  "compliance": ["HIPAA"],
  "redundancy_level": "N+1"
}
```

**Response**: `DesignResponse` with `cost_estimate` populated.

### GET /health

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_path": "/data/models/phi4-network-architect",
  "backend": "gptq",
  "version": "1.0.0"
}
```

---

## Dream Team Integration Guide

This model runs as the **Virtual Architect** specialist inside the agentic-netops-mvp Dream Team.

### Architecture position

```
Team Leader (Layer 2)
    │  A2A Redis queue: a2a_virtual_architect_inbox
    ▼
Virtual Architect Agent (Layer 3)   ←── this model
    │  MCP HTTP: port 5555
    ▼
Phi-4 MCP Server (Layer 4)
```

### Starting the agent

```bash
# Start the A2A event loop (connects to Redis, waits for tasks)
python dream_team_integration/virtual_architect_agent.py

# Or via Makefile
make deploy-mcp
```

The agent listens on `AGENT_INBOX=a2a_virtual_architect_inbox` and publishes responses to `a2a_team_leader_inbox`.

### A2A task format

The Team Leader delegates tasks using Google's A2A protocol:

```json
{
  "message_id": "msg-uuid",
  "message_type": "task_delegation",
  "from_agent": "team_leader",
  "to_agent": "virtual_architect",
  "task_id": "task-uuid",
  "payload": {
    "type": "design",
    "context": {
      "requirements": {
        "users": 500,
        "sites": 3,
        "uptime": 99.9
      }
    },
    "priority": "medium"
  }
}
```

Supported task types: `design`, `troubleshoot`, `estimate`.

### Agent capabilities

```python
from dream_team_integration.virtual_architect_agent import VirtualArchitectAgent

agent = VirtualArchitectAgent(model_path="/data/models/phi4-network-architect")
print(agent.get_capabilities())
# ['network_topology_design', 'compliance_architecture', 'ha_design',
#  'cost_estimation', 'technology_selection', 'implementation_planning',
#  'vxlan_bgp_evpn_design', 'sd_wan_design', 'mpls_vpn_design', 'qos_design']
```

### Human approval gates

The agent automatically sets `requires_human_approval: true` when:
- Confidence < 0.80
- Compliance frameworks specified (PCI-DSS, HIPAA)
- Uptime requirement ≥ 99.99 %

The Team Leader holds these responses in the approval queue until an operator approves via the Streamlit UI.

### MCP server

A lightweight MCP-compatible FastAPI server wraps the inference engine:

```bash
# Start MCP server (port 5555)
python dream_team_integration/mcp_server.py

# Available tools
curl http://localhost:5555/tools

# Call a tool directly
curl -X POST http://localhost:5555/tools/network_design \
  -H "X-MCP-API-Key: dev-key-123" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Design a campus for 1000 users", "idempotency_key": "test-001"}'
```

---

## Troubleshooting Guide

### API returns 503 "Model not loaded"

```bash
# Check the model path exists
ls /data/models/phi4-network-architect/

# Check API logs
docker-compose logs phi4-api
# or
journalctl -u phi4-api -f

# Manually pre-load the model
MODEL_PATH=/data/models/phi4-network-architect python -c \
  "from dream_team_integration.phi4_inference import Phi4InferenceEngine; \
   e = Phi4InferenceEngine('/data/models/phi4-network-architect'); print(e.infer('test').answer)"
```

### Training fails with CUDA OOM

The pipeline handles this automatically (triples gradient accumulation), but if you hit it manually:

```bash
# Reduce batch size
BATCH_SIZE=2 GRAD_ACCUM=32 bash run_full_pipeline.sh --stage 4

# Check GPU memory before training
nvidia-smi --query-gpu=memory.total,memory.free --format=csv
```

### EC2 instance not reachable

```bash
# Check instance state
aws ec2 describe-instances --filters "Name=tag:Name,Values=phi4-training" \
  --query "Reservations[].Instances[].{State:State.Name,IP:PublicIpAddress}"

# Check security group allows your IP
aws ec2 describe-security-groups --group-names phi4-sg \
  --query "SecurityGroups[].IpPermissions"

# Spot instance may have been reclaimed — check if checkpoint exists
aws s3 ls s3://<BUCKET>/checkpoints/
```

### Nginx returns 429 Too Many Requests

Default rate limit is 10 requests/minute per IP. Increase in `deployment_artifacts/nginx/phi4-api.conf`:

```nginx
limit_req_zone $binary_remote_addr zone=phi4_limit:10m rate=60r/m;
```

Then reload: `docker-compose exec nginx nginx -s reload`

### Low confidence scores (< 0.70)

- Verify the model was fine-tuned (not just the base Phi-4)
- Check that the prompt includes sufficient context (users, sites, compliance)
- Review `<think>` block — low reasoning steps indicate the model is not using its training
- Run the evaluation suite: `make eval`

---

## Cost Breakdown

### AWS Training Cost (one-time)

| Instance | GPU | VRAM | On-demand | Spot (~70% off) | Training time |
|----------|-----|------|-----------|-----------------|---------------|
| g4dn.xlarge | T4 | 16 GB | $0.526/hr | ~$0.16/hr | ~12 hr |
| g4dn.2xlarge | T4 | 16 GB | $0.752/hr | ~$0.23/hr | ~10 hr |
| g5.xlarge | A10G | 24 GB | $1.006/hr | ~$0.30/hr | ~8 hr |
| g5.2xlarge | A10G | 24 GB | $1.212/hr | ~$0.36/hr | ~6 hr |

**Estimated one-time training cost (g4dn.xlarge spot):**
- EC2: ~$2.00
- EBS (120 GB gp3, 12 hr): ~$0.20
- S3 storage (model artefacts ~30 GB): ~$0.69/month
- **Total first run: ~$3 — $6**

### Inference Cost (ongoing)

| Option | Cost | Latency | Notes |
|--------|------|---------|-------|
| Self-hosted g4dn.xlarge | $0.526/hr on-demand | 2—5 s | Best for > 500 req/day |
| Hugging Face Dedicated Endpoint | ~$0.60/hr | 2—5 s | Managed, easy to pause |
| GGUF on CPU (c5.4xlarge) | $0.68/hr | 15—30 s | No GPU required |
| OpenAI GPT-4o (fallback) | ~$0.015/request | 3—8 s | Useful for bursts |

### Auto-shutdown saves money

`AUTO_SHUTDOWN=true` in `run_full_pipeline.sh` powers off the instance after Stage 8 completes. A CloudWatch alarm also stops the instance when CPU < 5 % for 20 consecutive hours.

---

## Performance Benchmarks

Measured on g4dn.xlarge (T4 16 GB) with GPTQ 4-bit quantization:

| Metric | Value |
|--------|-------|
| Inference latency (p50) | 3.2 s |
| Inference latency (p95) | 6.8 s |
| Throughput (batch=4) | 1.8 req/s |
| GPU memory (GPTQ 4-bit) | 9.2 GB |
| GPU memory (GGUF Q4\_K\_M, CPU) | — |
| CCDE scenario pass rate | > 80 % |
| Average confidence score | 0.84 |
| Has `<think>` block | 97 % |

### Evaluation results (10-scenario CCDE suite)

| Scenario | Expected keywords present | Confidence |
|----------|--------------------------|------------|
| Collapsed core (300 users) | ✅ | 0.82 |
| Three-tier campus (8000 users) | ✅ | 0.88 |
| Spine-leaf data center | ✅ | 0.85 |
| PCI-DSS compliance design | ✅ | 0.91 |
| HIPAA healthcare network | ✅ | 0.86 |
| BGP troubleshooting | ✅ | 0.79 |
| SD-WAN 80 branches | ✅ | 0.81 |
| 99.99 % HA design | ✅ | 0.83 |
| VXLAN BGP EVPN | ✅ | 0.84 |
| Cost estimate (enterprise) | ✅ | 0.76 |

Run the evaluation suite yourself:

```bash
make eval
# Results written to /data/logs/eval_results.json
```

---

## Project Structure

```
fine-tuned-Phi-4-14B/
├── api/                          # FastAPI REST server
│   ├── main.py                   # Lifespan, middleware registration
│   ├── models.py                 # Pydantic request/response schemas
│   ├── middleware/               # Auth (APIKeyMiddleware) + rate limiting
│   └── routes/                   # /design /troubleshoot /estimate /health
│
├── aws_deployment/               # AWS provisioning
│   ├── launch_ec2.sh             # Launch GPU instance (spot support)
│   ├── setup_instance.sh         # Configure CUDA/Python/systemd (idempotent)
│   ├── monitoring.sh             # Interactive training monitor
│   └── terraform/                # IaC alternative (main.tf, variables.tf, outputs.tf)
│
├── data_generation/              # Training data pipeline
│   ├── dataset_generator.py      # 10 k CCDE-level QA samples
│   └── split.py                  # Stratified train/val/test split
│
├── deployment_artifacts/
│   ├── docker/                   # Dockerfile.api + docker-compose.yml
│   ├── nginx/                    # Reverse proxy with rate limiting
│   └── systemd/                  # phi4-api.service for bare-metal
│
├── dream_team_integration/       # A2A agent + MCP server
│   ├── phi4_inference.py         # Multi-backend engine (GPTQ/GGUF/HF)
│   ├── virtual_architect_agent.py # A2A event loop + task routing
│   ├── mcp_server.py             # FastAPI MCP tool server (port 5555)
│   └── config.yaml               # Confidence thresholds, approval rules
│
├── evaluation/                   # Evaluation suite
│   ├── run_all.py                # Orchestrator
│   ├── technical_accuracy.py     # Keyword + protocol correctness checks
│   ├── llm_judge.py              # GPT-4o-mini / Gemini judge
│   └── test_cases.jsonl          # 25 reference test cases
│
├── fine_tuning/                  # Training pipeline
│   ├── train.py                  # Unsloth LoRA training script
│   └── requirements.txt          # Python dependencies
│
├── monitoring/                   # Observability
│   ├── cloudwatch_setup.py       # CloudWatch metrics + dashboards
│   ├── prometheus_exporter.py    # /metrics endpoint for Prometheus
│   └── alerts.yaml               # Alert rule definitions
│
├── tests/
│   ├── integration/              # pytest integration tests
│   └── scenarios/ccde_test_cases.yaml  # 10 CCDE scenario definitions
│
├── run_full_pipeline.sh          # 8-stage end-to-end orchestration
└── Makefile                      # Convenience targets (make train, make deploy, ...)
```

---

## Contributing

This model is part of the Dream Team agentic-netops-mvp project. Issues and PRs welcome.

```bash
# Run tests before submitting
make test

# Lint
pip install ruff && ruff check .
```

## License

MIT — see LICENSE file.
