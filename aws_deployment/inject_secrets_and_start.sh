#!/usr/bin/env bash
# =============================================================================
# inject_secrets_and_start.sh — SSH into the EC2 instance, inject API keys,
# wait for setup_instance.sh to complete, then start 3-stage training.
#
# Usage:
#   bash aws_deployment/inject_secrets_and_start.sh <public-ip>
#
# Prerequisites: SSH key at ~/.ssh/phi4-training-key.pem
# =============================================================================
set -euo pipefail

PUBLIC_IP="${1:-}"
[[ -n "$PUBLIC_IP" ]] || { echo "Usage: $0 <public-ip>"; exit 1; }

KEY_FILE="${HOME}/.ssh/phi4-training-key.pem"
SSH="ssh -i $KEY_FILE -o StrictHostKeyChecking=no -o ConnectTimeout=10 ubuntu@$PUBLIC_IP"
SCP="scp -i $KEY_FILE -o StrictHostKeyChecking=no"

# Read from environment (set in .env or export before running)
OPENAI_API_KEY="${OPENAI_API_KEY:-}"
HF_TOKEN="${HUGGINGFACE_TOKEN:-${HF_TOKEN:-}}"

[[ -n "$OPENAI_API_KEY" ]] || { echo "ERROR: OPENAI_API_KEY not set. Export it before running."; exit 1; }
[[ -n "$HF_TOKEN" ]] || { echo "ERROR: HUGGINGFACE_TOKEN not set. Export it before running."; exit 1; }

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ─────────────────────────────────────────────────────────────────────────────
# 1. Wait for SSH to be available
# ─────────────────────────────────────────────────────────────────────────────
log "Waiting for SSH on $PUBLIC_IP..."
for i in $(seq 1 30); do
    if $SSH "echo ok" &>/dev/null; then
        log "  SSH ready"
        break
    fi
    log "  Attempt $i/30 — waiting 10s..."
    sleep 10
done

# ─────────────────────────────────────────────────────────────────────────────
# 2. Wait for setup_instance.sh to complete (cloud-init)
# ─────────────────────────────────────────────────────────────────────────────
log "Waiting for setup_instance.sh to complete (30-60 min for flash-attn)..."
log "  You can monitor with: $SSH 'tail -f /var/log/phi4-setup.log'"
for i in $(seq 1 120); do
    if $SSH "grep -q 'Setup Complete' /var/log/phi4-setup.log 2>/dev/null"; then
        log "  Setup complete!"
        break
    fi
    # Check for errors
    if $SSH "grep -q 'ERROR\|FAILED\|Traceback' /var/log/phi4-setup.log 2>/dev/null"; then
        log "  WARNING: Possible errors in setup log. Check: $SSH 'tail -50 /var/log/phi4-setup.log'"
    fi
    if [[ $i -eq 120 ]]; then
        log "  TIMEOUT: Setup took > 20 minutes. Proceeding anyway..."
    fi
    log "  [$i/120] Still setting up... ($(( i * 10 / 60 ))m elapsed)"
    sleep 10
done

# ─────────────────────────────────────────────────────────────────────────────
# 3. Inject API keys into .env
# ─────────────────────────────────────────────────────────────────────────────
log "Injecting API keys..."
$SSH "
    REPO_DIR=\$HOME/fine-tuned-Phi-4-14B
    ENV_FILE=\$REPO_DIR/.env

    # Update or append OPENAI_API_KEY
    if grep -q '^OPENAI_API_KEY=' \"\$ENV_FILE\"; then
        sed -i 's|^OPENAI_API_KEY=.*|OPENAI_API_KEY=${OPENAI_API_KEY}|' \"\$ENV_FILE\"
    else
        echo 'OPENAI_API_KEY=${OPENAI_API_KEY}' >> \"\$ENV_FILE\"
    fi

    # Update or append HUGGINGFACE_TOKEN
    if grep -q '^HUGGINGFACE_TOKEN=' \"\$ENV_FILE\"; then
        sed -i 's|^HUGGINGFACE_TOKEN=.*|HUGGINGFACE_TOKEN=${HF_TOKEN}|' \"\$ENV_FILE\"
    else
        echo 'HUGGINGFACE_TOKEN=${HF_TOKEN}' >> \"\$ENV_FILE\"
    fi

    # Also login to huggingface-hub so model downloads work
    source \$HOME/phi4-env/bin/activate
    python -c \"from huggingface_hub import login; login(token='${HF_TOKEN}')\" || true

    echo 'Keys injected: OPENAI_API_KEY and HUGGINGFACE_TOKEN'
    grep -E 'OPENAI_API_KEY=|HUGGINGFACE_TOKEN=' \"\$ENV_FILE\" | sed 's/=.*/=<REDACTED>/'
"
log "  Keys injected successfully"

# ─────────────────────────────────────────────────────────────────────────────
# 4. Start training pipeline in a screen session
# ─────────────────────────────────────────────────────────────────────────────
log "Starting 3-stage training pipeline in screen session 'training'..."
$SSH "
    REPO_DIR=\$HOME/fine-tuned-Phi-4-14B
    source \$HOME/phi4-env/bin/activate
    source \$REPO_DIR/.env
    export HUGGINGFACE_TOKEN=${HF_TOKEN}
    export OPENAI_API_KEY=${OPENAI_API_KEY}

    # Start training in detached screen session
    screen -dmS training bash -c '
        cd \$REPO_DIR
        source \$HOME/phi4-env/bin/activate
        source .env
        export HUGGINGFACE_TOKEN=${HF_TOKEN}
        export OPENAI_API_KEY=${OPENAI_API_KEY}
        echo \"=== Training started \$(date) ==\" | tee /data/logs/pipeline.log
        bash run_all_stages.sh 2>&1 | tee -a /data/logs/pipeline.log
        echo \"=== Training complete \$(date) ==\" | tee -a /data/logs/pipeline.log
    '
    echo \"Screen session started. Training is running.\"
    screen -ls
"

# ─────────────────────────────────────────────────────────────────────────────
# 5. Print monitoring commands
# ─────────────────────────────────────────────────────────────────────────────
log ""
log "=== Training Started ==="
log ""
log "Monitor training:"
log "  ssh -i $KEY_FILE ubuntu@$PUBLIC_IP"
log "  screen -r training          # attach to training session (Ctrl+A D to detach)"
log "  tail -f /data/logs/pipeline.log"
log "  watch -n5 nvidia-smi        # GPU utilization"
log ""
log "Stage evaluation (run after each stage):"
log "  python evaluation/stage_tracker.py --stage 1 --model /data/merged/stage1"
log "  python evaluation/compare_models.py"
log ""
log "TensorBoard tunnel:"
log "  ssh -i $KEY_FILE -L 6006:localhost:6006 ubuntu@$PUBLIC_IP"
log "  # Then open http://localhost:6006"
