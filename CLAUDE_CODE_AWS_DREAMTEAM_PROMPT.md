# Claude Code Prompt: Deploy & Fine-Tune Phi-4 on AWS EC2 for Dream Team Integration

## Context

I need to deploy the Phi-4 network architect fine-tuning pipeline on AWS EC2, train the model, and prepare it for integration with my Dream Team multi-agent system (VexpertAI).

**GitHub Repo:** https://github.com/eduardd76/fine-tuned-Phi-4-14B  
**Target System:** Dream Team (VexpertAI) - Multi-agent network operations system  
**Current Agents:** Team Leader (Mistral-7B), Stability (Qwen-2.5-7B), Security, Troubleshooting (CodeLlama-7B)  
**New Agent:** Virtual Architect (Phi-4-14B fine-tuned)

## Objective

Create a complete deployment package that:
1. Launches and configures AWS EC2 instance
2. Clones repo and sets up environment
3. Runs fine-tuning pipeline (dataset generation → training → quantization)
4. Creates integration module for Dream Team
5. Provides API endpoint or local inference interface
6. Includes monitoring, checkpointing, and recovery

## AWS Resources Available

- **Credits:** Free tier available
- **Preferred Region:** us-east-1 (or specify your region)
- **Instance Type:** g4dn.xlarge (T4 GPU, 16GB VRAM) or g5.xlarge (A10G, 24GB VRAM)
- **Storage:** 100GB+ EBS volume

## Deliverables Required

### 1. AWS Deployment Scripts

Create `aws_deployment/` directory with:

```
aws_deployment/
├── launch_ec2.sh              # Launch EC2 with proper config
├── setup_instance.sh          # Install all dependencies on EC2
├── cloudformation.yaml        # Infrastructure as Code (optional)
├── terraform/                 # Alternative IaC (optional)
│   ├── main.tf
│   ├── variables.tf
│   └── outputs.tf
└── monitoring.sh              # CloudWatch metrics setup
```

**launch_ec2.sh must:**
- Launch g4dn.xlarge or g5.xlarge instance
- Use Deep Learning AMI (Ubuntu 22.04)
- Configure security groups (SSH, optional API port)
- Attach 100GB EBS volume
- Set up CloudWatch monitoring
- Output connection details

**setup_instance.sh must:**
- Update system packages
- Install CUDA drivers (if not in AMI)
- Install Python 3.10+, pip, virtualenv
- Install Git, tmux, screen
- Clone the fine-tuned-Phi-4-14B repo
- Create Python virtual environment
- Install all requirements (PyTorch, Unsloth, Transformers, etc.)
- Set environment variables
- Create systemd service for auto-restart on failure

### 2. Pipeline Execution Script

Create `run_full_pipeline.sh`:

```bash
#!/bin/bash
# Complete pipeline execution with error handling, logging, and checkpointing

# Required environment variables:
# - OPENAI_API_KEY: For dataset generation reasoning validation
# - HUGGINGFACE_TOKEN: For model downloads (optional)
# - AWS_REGION: For S3 backup (optional)

# Steps:
# 1. Knowledge extraction validation
# 2. Dataset generation (10,000 samples, ~2 hours)
# 3. Dataset validation and quality checks
# 4. Train/val/test split
# 5. Fine-tuning (12 hours, checkpoints every 500 steps)
# 6. Model evaluation
# 7. Quantization (GPTQ for GPU, GGUF for CPU)
# 8. Upload to S3 or HuggingFace Hub
# 9. Create Dream Team integration module
```

**Features required:**
- Automatic checkpointing (resume on failure)
- Progress logging to CloudWatch
- Email/SNS notification on completion
- Automatic shutdown on completion (save credits)
- S3 backup of generated data and checkpoints
- Error recovery and retry logic

### 3. Dream Team Integration Module

Create `dream_team_integration/` with:

```
dream_team_integration/
├── virtual_architect_agent.py     # Main agent class
├── phi4_inference.py              # Inference wrapper
├── mcp_server.py                  # MCP protocol server
├── a2a_connector.py               # Agent-to-agent communication
├── config.yaml                    # Integration configuration
├── requirements.txt
└── test_integration.py            # Integration tests
```

**virtual_architect_agent.py must:**
- Implement same interface as other Dream Team agents
- Accept network design requests via MCP/A2A
- Return responses with reasoning chains (`<think>` tags)
- Support streaming responses
- Handle multi-turn conversations
- Integrate with Team Leader coordination
- Log decisions for audit trail

**phi4_inference.py must:**
- Load quantized Phi-4 model (GPTQ or GGUF)
- Handle `<think>` tag extraction
- Parse reasoning chains
- Extract final recommendations
- Measure inference latency
- Support batch processing
- Cache frequent queries

**mcp_server.py must:**
- Expose network architecture design endpoint
- Expose troubleshooting endpoint
- Expose cost estimation endpoint
- Handle concurrent requests
- Rate limiting
- Authentication (API key)
- Request/response logging

### 4. API Interface (Optional but Recommended)

Create `api/` with FastAPI server:

```
api/
├── main.py                    # FastAPI app
├── models.py                  # Pydantic models
├── routes/
│   ├── design.py             # POST /api/v1/design
│   ├── troubleshoot.py       # POST /api/v1/troubleshoot
│   ├── estimate.py           # POST /api/v1/estimate
│   └── health.py             # GET /api/v1/health
├── middleware/
│   ├── auth.py               # API key validation
│   └── rate_limit.py         # Rate limiting
└── docker/
    ├── Dockerfile
    └── docker-compose.yml
```

**API Endpoints:**

```python
POST /api/v1/design
{
  "requirements": {
    "users": 5000,
    "sites": 20,
    "compliance": ["PCI-DSS", "HIPAA"],
    "uptime": 99.99,
    "budget": "flexible"
  }
}

Response:
{
  "reasoning": "<think>Step 1: ...</think>",
  "design": "Three-tier architecture...",
  "cost_estimate": {
    "capex": 9300000,
    "annual_opex": 1860000
  },
  "sources": ["Enterprise Design Ch3", "PCI-DSS 4.0.1"],
  "confidence": 0.95
}
```

### 5. Monitoring and Observability

Create `monitoring/`:

```
monitoring/
├── cloudwatch_setup.py        # CloudWatch metrics
├── prometheus_exporter.py     # Prometheus metrics
├── grafana_dashboard.json     # Grafana dashboard config
└── alerts.yaml               # Alert rules
```

**Metrics to track:**
- Training loss (per step)
- Validation accuracy
- Inference latency (p50, p95, p99)
- GPU utilization
- Memory usage
- Request rate
- Error rate
- Model cache hit rate

### 6. Deployment Artifacts

Create `deployment_artifacts/`:

```
deployment_artifacts/
├── systemd/
│   ├── phi4-api.service      # Systemd service for API
│   └── phi4-worker.service   # Background worker
├── nginx/
│   └── phi4-api.conf         # Nginx reverse proxy config
├── docker/
│   ├── Dockerfile.api        # API container
│   ├── Dockerfile.worker     # Worker container
│   └── docker-compose.yml
└── kubernetes/               # If scaling needed
    ├── deployment.yaml
    ├── service.yaml
    └── ingress.yaml
```

### 7. Testing Suite

Create `tests/`:

```
tests/
├── unit/
│   ├── test_inference.py
│   ├── test_reasoning_extraction.py
│   └── test_integration_module.py
├── integration/
│   ├── test_dream_team_communication.py
│   ├── test_api_endpoints.py
│   └── test_mcp_server.py
├── performance/
│   ├── test_latency.py
│   └── test_throughput.py
└── scenarios/
    └── ccde_test_cases.yaml   # 25 CCDE-level scenarios
```

## Specific Integration Requirements for Dream Team

### Agent Interface Compatibility

The Virtual Architect agent must match this interface:

```python
class VirtualArchitectAgent:
    """
    Phi-4 based network architecture agent
    Compatible with Dream Team multi-agent system
    """
    
    def __init__(self, model_path: str, config: dict):
        """Initialize with quantized Phi-4 model"""
        pass
    
    async def receive_task(self, task: dict) -> dict:
        """
        Receive task from Team Leader
        
        Args:
            task = {
                "type": "design" | "troubleshoot" | "estimate",
                "context": {...},
                "priority": "high" | "medium" | "low",
                "requestor": "team_leader" | "security_agent" | ...,
            }
        
        Returns:
            {
                "status": "success" | "error",
                "reasoning": "<think>...</think>",
                "recommendation": "...",
                "confidence": 0.0-1.0,
                "sources": [...],
                "estimated_time": "...",
                "cost_estimate": {...}
            }
        """
        pass
    
    async def collaborate(self, agent_id: str, question: str) -> dict:
        """Ask another agent for information"""
        pass
    
    def get_capabilities(self) -> list:
        """Return list of capabilities"""
        return [
            "network_topology_design",
            "compliance_architecture",
            "ha_design",
            "cost_estimation",
            "technology_selection",
            "implementation_planning"
        ]
```

### Communication Protocols

Support these protocols (match existing Dream Team):

1. **MCP (Model Context Protocol)**: For tool use and context sharing
2. **A2A (Agent-to-Agent)**: Via gRPC for low-latency communication
3. **ACP (Agent Control Protocol)**: For human oversight and approval

### Data Flow Integration

```
┌─────────────────┐
│   Team Leader   │  (Mistral-7B)
│   (Coordinator) │
└────────┬────────┘
         │
         ├─────────┐
         │         │
    ┌────▼────┐ ┌─▼──────────────┐
    │ Virtual │ │  Troubleshoot  │
    │Architect│ │     Agent      │
    │ (Phi-4) │ │  (CodeLlama)   │
    └────┬────┘ └─┬──────────────┘
         │        │
         ├────────┴──────────┐
         │                   │
    ┌────▼─────┐      ┌─────▼──────┐
    │ Stability│      │  Security  │
    │  Agent   │      │   Agent    │
    │ (Qwen)   │      │            │
    └──────────┘      └────────────┘
```

## Environment Variables Required

```bash
# Required
OPENAI_API_KEY=sk-...              # For dataset generation LLM-as-judge
HUGGINGFACE_TOKEN=hf_...           # For model downloads

# Optional
AWS_ACCESS_KEY_ID=...              # For S3 backup
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=us-east-1
S3_BUCKET=phi4-training-artifacts

# Dream Team Integration
DREAM_TEAM_API_URL=http://localhost:8000
MCP_SERVER_PORT=5555
A2A_GRPC_PORT=50051
```

## Success Criteria

After deployment and training, I should be able to:

1. ✅ SSH into EC2 and see training in progress
2. ✅ View training metrics in CloudWatch
3. ✅ Access API endpoint: `curl http://ec2-ip:8000/api/v1/health`
4. ✅ Send design request and get response with reasoning
5. ✅ Integrate with Dream Team and have agents communicate
6. ✅ Run CCDE test scenarios and get >95% accuracy
7. ✅ Deploy quantized model (<10GB, <3s inference)
8. ✅ Auto-shutdown EC2 after training (save credits)

## Execution Flow

```bash
# On local machine
cd aws_deployment
./launch_ec2.sh

# SSH into EC2 (script outputs connection details)
ssh -i key.pem ubuntu@ec2-xx-xx-xx-xx.compute.amazonaws.com

# On EC2 instance (setup script auto-runs, or manual)
cd ~
git clone https://github.com/eduardd76/fine-tuned-Phi-4-14B.git
cd fine-tuned-Phi-4-14B
./aws_deployment/setup_instance.sh

# Set environment variables
export OPENAI_API_KEY=sk-...
export HUGGINGFACE_TOKEN=hf_...

# Run pipeline in screen/tmux (auto-resumes on disconnect)
screen -S training
./run_full_pipeline.sh

# Detach: Ctrl+A then D
# Reattach: screen -r training

# Monitor from local machine
./aws_deployment/monitoring.sh ec2-xx-xx-xx-xx

# After training (12-15 hours)
# Download trained model
scp -i key.pem -r ubuntu@ec2-ip:~/fine-tuned-Phi-4-14B/models ./

# Or deploy API directly on EC2
./deployment_artifacts/docker/docker-compose up -d

# Test integration
python dream_team_integration/test_integration.py
```

## Cost Optimization

**Must include:**
- Spot instance support (50-70% cheaper)
- Auto-shutdown after training completion
- S3 lifecycle policies (move to Glacier after 30 days)
- EBS snapshot before termination
- Cost tracking script

```bash
# Estimated costs (with free credits = $0)
# Without credits:
# - g4dn.xlarge: $0.526/hour × 15 hours = $7.89
# - Storage: $0.10/GB × 100GB = $10/month
# - Total: ~$18 for complete training
```

## Error Handling

**Must handle:**
- CUDA out of memory → reduce batch size, retry
- EC2 spot interruption → checkpoint, resume on new instance
- Network timeout → retry with exponential backoff
- Dataset generation failure → skip invalid samples, continue
- Training divergence → restore last good checkpoint
- Disk full → compress logs, cleanup temp files

## Documentation Required

Create comprehensive `README.md` with:

1. Quick start (5 minutes to launch)
2. Detailed setup guide
3. Configuration options
4. API documentation
5. Dream Team integration guide
6. Troubleshooting guide
7. Cost breakdown
8. Performance benchmarks

## Output Format

All files should be:
- ✅ Production-ready (not templates or placeholders)
- ✅ Fully commented with explanations
- ✅ Error handling and logging
- ✅ Type hints (Python 3.10+)
- ✅ Security best practices (no hardcoded secrets)
- ✅ Tested and verified

## Additional Requirements

### Security
- Use AWS Secrets Manager for API keys
- Enable CloudWatch Logs encryption
- Restrict security groups (SSH from your IP only)
- Use IAM roles instead of access keys where possible
- Enable EBS encryption

### Backup
- Auto-backup checkpoints to S3 every hour
- Backup final model to S3 and HuggingFace Hub
- Snapshot EBS volume after training
- Export logs to S3 before instance termination

### Monitoring
- CloudWatch dashboard with key metrics
- SNS alerts on errors or completion
- Slack/Discord webhook notifications (optional)
- Training progress via TensorBoard (accessible via SSH tunnel)

## Dream Team Specific Integration Points

### 1. Shared Knowledge Base
Virtual Architect should access the same NotebookLM knowledge base used by other agents:

```python
# knowledge_base/notebooklm_client.py
class SharedKnowledgeBase:
    """Shared access to NotebookLM across all Dream Team agents"""
    def query(self, question: str) -> dict:
        """Query NotebookLM, cache results"""
        pass
```

### 2. Decision Logging
All recommendations must be logged for audit:

```python
# logging/decision_logger.py
class DecisionLogger:
    """Log all agent decisions for compliance and debugging"""
    def log_decision(self, agent: str, decision: dict):
        """Store in PostgreSQL/MongoDB with full reasoning chain"""
        pass
```

### 3. Human-in-the-Loop
Critical decisions require human approval:

```python
# approval/human_approval.py
class ApprovalWorkflow:
    """Queue high-stakes decisions for human review"""
    def request_approval(self, decision: dict) -> bool:
        """Send to approval queue, wait for human OK"""
        pass
```

## Final Deliverables Checklist

- [ ] AWS deployment scripts (launch, setup, monitoring)
- [ ] Full pipeline execution script with checkpointing
- [ ] Dream Team integration module (agent class, MCP server, A2A)
- [ ] API server (FastAPI with endpoints)
- [ ] Monitoring setup (CloudWatch, Prometheus, Grafana)
- [ ] Deployment artifacts (systemd, Docker, Nginx)
- [ ] Testing suite (unit, integration, performance)
- [ ] Documentation (README, API docs, integration guide)
- [ ] Cost optimization (spot instances, auto-shutdown)
- [ ] Security hardening (secrets, encryption, IAM roles)

## Timeline Expectations

- AWS setup: 30 minutes
- Dataset generation: 2 hours
- Training: 12 hours
- Evaluation: 1 hour
- Quantization: 1 hour
- Integration: 2 hours
- Testing: 1 hour
- **Total: ~20 hours** (mostly unattended)

## Success Validation

Run this command after deployment:

```bash
# Test end-to-end integration
curl -X POST http://ec2-ip:8000/api/v1/design \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{
    "requirements": {
      "users": 5000,
      "sites": 20,
      "compliance": ["PCI-DSS"],
      "uptime": 99.99
    }
  }'

# Should return:
# {
#   "reasoning": "<think>Step 1: Analyze scale...",
#   "design": "Three-tier architecture...",
#   "sources": ["Enterprise Design Ch3", "PCI-DSS 4.0.1"],
#   "confidence": 0.95
# }
```

---

**Build this complete deployment and integration package. Make it production-ready, secure, cost-optimized, and fully integrated with Dream Team.**
