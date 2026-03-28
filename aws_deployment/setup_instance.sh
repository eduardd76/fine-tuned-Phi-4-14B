#!/usr/bin/env bash
# =============================================================================
# setup_instance.sh — Configure EC2 for Phi-4 fine-tuning
# Target: AWS Deep Learning AMI (Ubuntu 22.04) with CUDA pre-installed
# Idempotent: safe to re-run
#
# Usage:
#   bash aws_deployment/setup_instance.sh
# =============================================================================
set -euo pipefail

LOG_FILE="/var/log/phi4-setup.log"
REPO_DIR="${HOME}/fine-tuned-Phi-4-14B"
VENV_DIR="${HOME}/phi4-env"
DATA_DIR="/data"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG_FILE"; }

log "=== Phi-4 Instance Setup ==="
log "  User: $USER"
log "  Home: $HOME"
log "  Repo: $REPO_DIR"

# ─────────────────────────────────────────────────────────────────────────────
# 1. System packages
# ─────────────────────────────────────────────────────────────────────────────
log "Step 1: System packages"
sudo apt-get update -qq
sudo apt-get install -y -qq \
    git tmux screen htop nvtop curl wget jq unzip \
    build-essential cmake \
    python3.11 python3.11-venv python3.11-dev python3-pip \
    libssl-dev libffi-dev \
    nginx \
    awscli

# ─────────────────────────────────────────────────────────────────────────────
# 2. Verify CUDA (DLAMI should have it)
# ─────────────────────────────────────────────────────────────────────────────
log "Step 2: CUDA verification"
if command -v nvcc &>/dev/null; then
    log "  CUDA: $(nvcc --version | grep release | awk '{print $5}' | tr -d ,)"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    log "  WARNING: nvcc not found. DLAMI should include CUDA."
    log "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'not detected')"
fi

# ─────────────────────────────────────────────────────────────────────────────
# 3. EBS data volume
# ─────────────────────────────────────────────────────────────────────────────
log "Step 3: Data volume setup"
if ! mountpoint -q "$DATA_DIR" 2>/dev/null; then
    # Find secondary EBS (not root)
    ROOT_DEV=$(df / | tail -1 | awk '{print $1}' | sed 's/p[0-9]$//')
    DATA_DEV=$(lsblk -dpno NAME,TYPE | awk '$2=="disk"' | grep -v "$(basename $ROOT_DEV)" | head -1 | awk '{print $1}')

    if [[ -n "$DATA_DEV" ]]; then
        if ! blkid "$DATA_DEV" &>/dev/null; then
            log "  Formatting $DATA_DEV..."
            sudo mkfs.ext4 -F "$DATA_DEV"
        fi
        sudo mkdir -p "$DATA_DIR"
        UUID=$(sudo blkid -s UUID -o value "$DATA_DEV")
        if ! grep -q "$UUID" /etc/fstab; then
            echo "UUID=$UUID $DATA_DIR ext4 defaults,nofail 0 2" | sudo tee -a /etc/fstab
        fi
        sudo mount "$DATA_DIR" || sudo mount -a
        sudo chown -R "$USER:$USER" "$DATA_DIR"
        log "  Mounted $DATA_DEV → $DATA_DIR"
    else
        sudo mkdir -p "$DATA_DIR"
        sudo chown -R "$USER:$USER" "$DATA_DIR"
        log "  No secondary EBS found — using local storage at $DATA_DIR"
    fi
fi

mkdir -p "$DATA_DIR"/{generated_data,models,checkpoints,logs,cache}
log "  Data directories created"

# ─────────────────────────────────────────────────────────────────────────────
# 4. Clone / update repo
# ─────────────────────────────────────────────────────────────────────────────
log "Step 4: Repository"
if [[ ! -d "$REPO_DIR" ]]; then
    log "  Cloning fine-tuned-Phi-4-14B..."
    git clone https://github.com/eduardd76/fine-tuned-Phi-4-14B.git "$REPO_DIR"
else
    log "  Pulling latest changes..."
    git -C "$REPO_DIR" pull --rebase
fi
log "  Repo ready: $REPO_DIR"

# ─────────────────────────────────────────────────────────────────────────────
# 5. Python virtual environment
# ─────────────────────────────────────────────────────────────────────────────
log "Step 5: Python virtualenv"
if [[ ! -d "$VENV_DIR" ]]; then
    python3.11 -m venv "$VENV_DIR"
    log "  Virtualenv created: $VENV_DIR"
else
    log "  Virtualenv exists: $VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
pip install --quiet --upgrade pip wheel setuptools

# ─────────────────────────────────────────────────────────────────────────────
# 6. Python packages
# ─────────────────────────────────────────────────────────────────────────────
log "Step 6: Python packages (this takes 5-10 minutes)"

# PyTorch for CUDA 12.1
pip install --quiet torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# Core ML
pip install --quiet \
    "unsloth[colab-new]>=2024.12.0" \
    "transformers>=4.47.0" \
    "trl>=0.13.0" \
    "peft>=0.14.0" \
    "bitsandbytes>=0.45.0" \
    "accelerate>=0.34.0" \
    "datasets>=3.0.0"

# Quantization
pip install --quiet \
    "auto-gptq>=0.7.1" \
    "optimum>=1.23.0"

# API and serving
pip install --quiet \
    "fastapi>=0.115.0" \
    "uvicorn[standard]>=0.32.0" \
    "httpx>=0.27.0" \
    "pydantic>=2.9.0"

# AWS, monitoring
pip install --quiet \
    "boto3>=1.35.0" \
    "psutil>=6.0.0" \
    "prometheus-client>=0.21.0"

# Utilities
pip install --quiet \
    "pyyaml>=6.0.0" \
    "tqdm>=4.66.0" \
    "openai>=1.50.0" \
    "huggingface-hub>=0.24.0" \
    "tiktoken>=0.7.0" \
    "sentencepiece>=0.2.0"

# Install project requirements
if [[ -f "$REPO_DIR/fine_tuning/requirements.txt" ]]; then
    pip install --quiet -r "$REPO_DIR/fine_tuning/requirements.txt" || true
fi

log "  Packages installed"

# Verify GPU/torch
python -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB')
"

# ─────────────────────────────────────────────────────────────────────────────
# 7. Environment file
# ─────────────────────────────────────────────────────────────────────────────
log "Step 7: Environment configuration"
ENV_FILE="$REPO_DIR/.env"
if [[ ! -f "$ENV_FILE" ]]; then
    cat > "$ENV_FILE" << 'EOF'
# Phi-4 Fine-Tuning Environment
# Fill in your actual values before running

# Required for dataset generation LLM validation
OPENAI_API_KEY=

# Required to download gated models from HuggingFace
HUGGINGFACE_TOKEN=

# Optional: S3 backup
AWS_DEFAULT_REGION=us-east-1
S3_BUCKET=

# Dream Team integration
DREAM_TEAM_API_URL=http://localhost:8000
MCP_SERVER_PORT=5555
PHI4_API_PORT=8000
PHI4_API_KEY=phi4-key-change-me

# Paths (auto-configured)
DATA_DIR=/data
MODEL_OUTPUT_DIR=/data/models/phi4-network-architect
CHECKPOINT_DIR=/data/checkpoints
LOG_DIR=/data/logs

# Training overrides (optional)
BATCH_SIZE=1
GRAD_ACCUM=16
MAX_STEPS=0
EOF
    log "  Created .env template: $ENV_FILE"
    log "  IMPORTANT: Edit .env and set OPENAI_API_KEY before running pipeline"
fi

# ─────────────────────────────────────────────────────────────────────────────
# 8. Shell profile
# ─────────────────────────────────────────────────────────────────────────────
log "Step 8: Shell profile"
PROFILE_BLOCK='
# Phi-4 Training Environment
export PATH="$HOME/phi4-env/bin:$PATH"
export DATA_DIR=/data
export HF_HOME=/data/cache/huggingface
export TRANSFORMERS_CACHE=/data/cache/huggingface/transformers
export REPO_DIR="$HOME/fine-tuned-Phi-4-14B"
alias phi4="cd $REPO_DIR && source $HOME/phi4-env/bin/activate"
alias logs="tail -f /data/logs/pipeline.log"
alias gpu="watch -n1 nvidia-smi"
'

if ! grep -q "Phi-4 Training Environment" ~/.bashrc; then
    echo "$PROFILE_BLOCK" >> ~/.bashrc
fi

# ─────────────────────────────────────────────────────────────────────────────
# 9. Systemd service for API server
# ─────────────────────────────────────────────────────────────────────────────
log "Step 9: Systemd service"
sudo tee /etc/systemd/system/phi4-api.service > /dev/null << EOF
[Unit]
Description=Phi-4 Network Architect API Server
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$REPO_DIR
Environment="PATH=$VENV_DIR/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin"
EnvironmentFile=-$REPO_DIR/.env
ExecStart=$VENV_DIR/bin/uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 1
Restart=on-failure
RestartSec=10
StandardOutput=append:/data/logs/phi4-api.log
StandardError=append:/data/logs/phi4-api-err.log

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
log "  Service registered (start after training: sudo systemctl enable --now phi4-api)"

# ─────────────────────────────────────────────────────────────────────────────
# 10. CloudWatch agent config
# ─────────────────────────────────────────────────────────────────────────────
log "Step 10: CloudWatch agent"
sudo mkdir -p /opt/aws/amazon-cloudwatch-agent/etc/
sudo tee /opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json > /dev/null << 'EOF'
{
  "agent": {"run_as_user": "root"},
  "logs": {
    "logs_collected": {
      "files": {
        "collect_list": [
          {"file_path": "/data/logs/pipeline.log", "log_group_name": "/phi4/training", "log_stream_name": "pipeline"},
          {"file_path": "/var/log/phi4-setup.log", "log_group_name": "/phi4/setup", "log_stream_name": "setup"},
          {"file_path": "/data/logs/phi4-api.log", "log_group_name": "/phi4/api", "log_stream_name": "api"}
        ]
      }
    }
  },
  "metrics": {
    "metrics_collected": {
      "gpu": {
        "measurement": ["utilization_gpu", "utilization_memory", "memory_used", "memory_total", "temperature_gpu"],
        "metrics_collection_interval": 60
      },
      "cpu": {"measurement": ["cpu_usage_active"], "metrics_collection_interval": 60},
      "mem": {"measurement": ["mem_used_percent"], "metrics_collection_interval": 60},
      "disk": {"measurement": ["disk_used_percent"], "resources": ["*"], "metrics_collection_interval": 300}
    }
  }
}
EOF

if command -v amazon-cloudwatch-agent-ctl &>/dev/null; then
    sudo amazon-cloudwatch-agent-ctl -a fetch-config -m ec2 \
        -s -c file:/opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json
    log "  CloudWatch agent started"
else
    log "  CloudWatch agent not installed (install: snap install amazon-cloudwatch-agent)"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Done
# ─────────────────────────────────────────────────────────────────────────────
log ""
log "=== Setup Complete ==="
log "Next steps:"
log "  1. Edit .env: nano $REPO_DIR/.env"
log "  2. Set OPENAI_API_KEY and HUGGINGFACE_TOKEN"
log "  3. source ~/.bashrc"
log "  4. cd $REPO_DIR && screen -S training"
log "  5. ./run_full_pipeline.sh"
log ""
log "Useful commands:"
log "  gpu      — watch GPU utilization"
log "  logs     — tail training log"
log "  phi4     — cd to repo + activate venv"
