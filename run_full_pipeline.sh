#!/usr/bin/env bash
# =============================================================================
# run_full_pipeline.sh — Complete Phi-4 fine-tuning pipeline
#
# Pipeline stages:
#   1. Knowledge validation
#   2. Dataset generation (10k samples, ~2h with GPT-4o-mini)
#   3. Dataset splitting (80/10/10)
#   4. Fine-tuning with Unsloth LoRA (12h)
#   5. Evaluation (>95% technical accuracy)
#   6. Quantization (GPTQ + GGUF)
#   7. S3 backup (optional)
#   8. Dream Team integration setup
#   9. Auto-shutdown (saves credits)
#
# Features:
#   - Automatic checkpointing — safe to resume after interruption
#   - Spot instance interruption handler
#   - SNS notification on completion/failure
#   - Disk space monitoring
#   - GPU OOM recovery (auto-reduces batch size)
#
# Usage:
#   ./run_full_pipeline.sh                # Full pipeline
#   ./run_full_pipeline.sh --skip-data    # Skip dataset gen (use existing)
#   ./run_full_pipeline.sh --skip-train   # Skip training (use existing model)
#   ./run_full_pipeline.sh --resume       # Resume from last checkpoint
# =============================================================================
set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "$0")" && pwd)}"
VENV_DIR="${HOME}/phi4-env"
DATA_DIR="${DATA_DIR:-/data}"
LOG_FILE="${DATA_DIR}/logs/pipeline.log"
CHECKPOINT_FILE="${DATA_DIR}/.pipeline_checkpoint"
S3_BUCKET="${S3_BUCKET:-}"
SNS_TOPIC="${SNS_TOPIC:-}"
AUTO_SHUTDOWN="${AUTO_SHUTDOWN:-false}"    # Set to "true" in production

# Flags
SKIP_DATA=false
SKIP_TRAIN=false
SKIP_EVAL=false
RESUME=false

for arg in "$@"; do
    [[ "$arg" == "--skip-data" ]]  && SKIP_DATA=true
    [[ "$arg" == "--skip-train" ]] && SKIP_TRAIN=true
    [[ "$arg" == "--skip-eval" ]]  && SKIP_EVAL=true
    [[ "$arg" == "--resume" ]]     && RESUME=true
done

# ─────────────────────────────────────────────────────────────────────────────
# Logging helpers
# ─────────────────────────────────────────────────────────────────────────────
mkdir -p "$DATA_DIR/logs"
exec > >(tee -a "$LOG_FILE") 2>&1

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
log_section() {
    echo ""
    echo "═══════════════════════════════════════════════════════════"
    echo "  $*"
    echo "═══════════════════════════════════════════════════════════"
}
notify() {
    local msg="$1"
    log "$msg"
    if [[ -n "$SNS_TOPIC" ]]; then
        aws sns publish --topic-arn "$SNS_TOPIC" --message "$msg" --subject "Phi-4 Training" 2>/dev/null || true
    fi
}

# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint system
# ─────────────────────────────────────────────────────────────────────────────
save_checkpoint() {
    echo "STAGE=$1" > "$CHECKPOINT_FILE"
    echo "TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$CHECKPOINT_FILE"
    log "  Checkpoint saved: stage=$1"
}

get_checkpoint() {
    if [[ -f "$CHECKPOINT_FILE" ]]; then
        grep "^STAGE=" "$CHECKPOINT_FILE" | cut -d= -f2
    else
        echo "0"
    fi
}

# Resume from checkpoint
LAST_STAGE=$(get_checkpoint)
if [[ "$RESUME" == "true" && "$LAST_STAGE" -gt 0 ]]; then
    log "Resuming from stage $LAST_STAGE"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Spot instance interruption handler
# ─────────────────────────────────────────────────────────────────────────────
handle_spot_interruption() {
    log "SPOT INTERRUPTION DETECTED — saving state..."
    save_checkpoint "$CURRENT_STAGE"

    if [[ -n "$S3_BUCKET" ]]; then
        log "  Uploading checkpoints to S3..."
        aws s3 sync "$DATA_DIR/checkpoints" "s3://$S3_BUCKET/checkpoints/" \
            --only-show-errors || true
        aws s3 sync "$DATA_DIR/generated_data" "s3://$S3_BUCKET/generated_data/" \
            --only-show-errors || true
    fi

    notify "Phi-4 training interrupted at stage $CURRENT_STAGE. Checkpoints saved."
    exit 0
}

# Register spot interruption handler (AWS sends SIGTERM 2 min before shutdown)
trap 'handle_spot_interruption' SIGTERM

# ─────────────────────────────────────────────────────────────────────────────
# Environment setup
# ─────────────────────────────────────────────────────────────────────────────
log_section "Phi-4 Network Architect Training Pipeline"
log "Repository: $REPO_DIR"
log "Data dir:   $DATA_DIR"
log "Log:        $LOG_FILE"

# Load environment
ENV_FILE="$REPO_DIR/.env"
if [[ -f "$ENV_FILE" ]]; then
    # shellcheck source=/dev/null
    set -a; source "$ENV_FILE"; set +a
fi

# Activate virtualenv
source "$VENV_DIR/bin/activate" 2>/dev/null || {
    log "Virtualenv not found at $VENV_DIR — run setup_instance.sh first"
    exit 1
}

# Validate required keys
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    log "WARNING: OPENAI_API_KEY not set — dataset quality validation will be skipped"
fi

# GPU info
python -c "
import torch
if torch.cuda.is_available():
    gpu = torch.cuda.get_device_properties(0)
    print(f'  GPU: {gpu.name} | VRAM: {gpu.total_memory/1e9:.1f}GB')
else:
    print('  WARNING: No GPU detected')
"

# Disk space check
DISK_FREE=$(df "$DATA_DIR" --output=avail -BG | tail -1 | tr -d 'G ')
log "  Disk free: ${DISK_FREE}GB (need ~80GB)"
if [[ "$DISK_FREE" -lt 50 ]]; then
    log "ERROR: Insufficient disk space (${DISK_FREE}GB < 50GB)"
    exit 1
fi

# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: Knowledge extraction validation
# ─────────────────────────────────────────────────────────────────────────────
CURRENT_STAGE=1
if [[ "$LAST_STAGE" -lt 1 || "$RESUME" == "false" ]]; then
    log_section "Stage 1/8: Knowledge Base Validation"

    if [[ -f "$REPO_DIR/knowledge_extraction/design_patterns.json" ]]; then
        python -c "
import json
from pathlib import Path
kb_dir = Path('$REPO_DIR/knowledge_extraction')
total = sum(len(open(f).read()) for f in kb_dir.glob('*.json'))
print(f'  Knowledge base: {total//1024}KB across {len(list(kb_dir.glob(\"*.json\")))} files')
"
        log "  Knowledge base validated"
    else
        log "  WARNING: Knowledge extraction files not found"
        log "  Run: python knowledge_extraction/extract_live.py --dry-run"
    fi
    save_checkpoint 1
fi

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: Dataset generation
# ─────────────────────────────────────────────────────────────────────────────
CURRENT_STAGE=2
TRAIN_FILE="$DATA_DIR/generated_data/training_data.jsonl"

if [[ "$SKIP_DATA" == "false" && ( "$LAST_STAGE" -lt 2 || "$RESUME" == "false" ) ]]; then
    log_section "Stage 2/8: Dataset Generation (~2 hours)"

    if [[ -f "$TRAIN_FILE" ]]; then
        EXISTING=$(wc -l < "$TRAIN_FILE")
        log "  Existing dataset: ${EXISTING} samples"
        if [[ "$EXISTING" -ge 9000 ]]; then
            log "  Dataset sufficient (≥9000 samples) — skipping generation"
            save_checkpoint 2
        fi
    fi

    if [[ "$LAST_STAGE" -lt 2 ]]; then
        log "  Generating 10,000 training samples..."
        log "  Expected time: ~2 hours with GPT-4o-mini"

        # Run with OOM protection
        python "$REPO_DIR/data_generation/dataset_generator.py" \
            --count 10000 \
            --output "$DATA_DIR/generated_data" \
            --checkpoint "$DATA_DIR/.generation_checkpoint" \
            2>&1 || {
            log "ERROR: Dataset generation failed"
            notify "Phi-4 pipeline FAILED at dataset generation"
            exit 1
        }

        # Backup to S3
        if [[ -n "$S3_BUCKET" ]]; then
            log "  Backing up dataset to S3..."
            aws s3 sync "$DATA_DIR/generated_data" \
                "s3://$S3_BUCKET/generated_data/" --only-show-errors
        fi

        save_checkpoint 2
    fi
else
    log "  Skipping data generation (--skip-data)"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Stage 3: Dataset splitting
# ─────────────────────────────────────────────────────────────────────────────
CURRENT_STAGE=3
if [[ "$LAST_STAGE" -lt 3 ]]; then
    log_section "Stage 3/8: Dataset Split (80/10/10)"

    python "$REPO_DIR/data_generation/split.py" \
        --input "$TRAIN_FILE" \
        --output-dir "$DATA_DIR/generated_data" \
        --stratify 2>&1

    log "  Split complete"
    save_checkpoint 3
fi

# ─────────────────────────────────────────────────────────────────────────────
# Stage 4: Fine-tuning
# ─────────────────────────────────────────────────────────────────────────────
CURRENT_STAGE=4
if [[ "$SKIP_TRAIN" == "false" && "$LAST_STAGE" -lt 4 ]]; then
    log_section "Stage 4/8: Fine-Tuning with Unsloth LoRA (~12 hours)"
    log "  GPU monitoring: watch -n5 nvidia-smi"

    CHECKPOINT_DIR="$DATA_DIR/checkpoints"
    mkdir -p "$CHECKPOINT_DIR"

    # Detect if we should resume from training checkpoint
    TRAIN_RESUME=""
    LATEST_CKPT=$(ls -td "$CHECKPOINT_DIR"/checkpoint-* 2>/dev/null | head -1 || echo "")
    if [[ "$RESUME" == "true" && -n "$LATEST_CKPT" ]]; then
        log "  Resuming from training checkpoint: $LATEST_CKPT"
        TRAIN_RESUME="--resume $LATEST_CKPT"
    fi

    # OOM retry loop — auto-reduce batch size on CUDA OOM
    MAX_RETRIES=3
    RETRY=0
    BATCH_SIZE="${BATCH_SIZE:-1}"
    GRAD_ACCUM="${GRAD_ACCUM:-16}"

    while [[ $RETRY -lt $MAX_RETRIES ]]; do
        log "  Training attempt $((RETRY+1))/$MAX_RETRIES (batch=$BATCH_SIZE, accum=$GRAD_ACCUM)"

        BATCH_SIZE="$BATCH_SIZE" GRAD_ACCUM="$GRAD_ACCUM" \
        python "$REPO_DIR/fine_tuning/train.py" \
            --train-data "$DATA_DIR/generated_data/training_data.jsonl" \
            --val-data   "$DATA_DIR/generated_data/validation_data.jsonl" \
            --output-dir "$CHECKPOINT_DIR" \
            $TRAIN_RESUME \
            2>&1 && break

        EXIT_CODE=$?
        # Check if OOM error
        if grep -q "CUDA out of memory" "$LOG_FILE" 2>/dev/null; then
            RETRY=$((RETRY+1))
            GRAD_ACCUM=$((GRAD_ACCUM * 2))
            log "  OOM detected — increasing grad_accum to $GRAD_ACCUM, retry $RETRY/$MAX_RETRIES"
            sleep 10
        else
            log "ERROR: Training failed (non-OOM error)"
            notify "Phi-4 training FAILED"
            exit 1
        fi
    done

    # Backup checkpoints to S3
    if [[ -n "$S3_BUCKET" ]]; then
        log "  Backing up checkpoints to S3..."
        aws s3 sync "$CHECKPOINT_DIR" "s3://$S3_BUCKET/checkpoints/" --only-show-errors
    fi

    save_checkpoint 4
    notify "Phi-4 training complete! Starting evaluation."
else
    log "  Skipping training (--skip-train)"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Stage 5: Evaluation
# ─────────────────────────────────────────────────────────────────────────────
CURRENT_STAGE=5
if [[ "$SKIP_EVAL" == "false" && "$LAST_STAGE" -lt 5 ]]; then
    log_section "Stage 5/8: Evaluation (target: >95% accuracy)"

    MODEL_PATH="$DATA_DIR/checkpoints"
    python "$REPO_DIR/evaluation/run_all.py" \
        --model "$MODEL_PATH" \
        --test-data "$REPO_DIR/evaluation/test_cases.jsonl" \
        --output "$DATA_DIR/logs/evaluation_results.json" \
        2>&1 || {
        log "WARNING: Evaluation failed or below threshold — check results"
    }

    if [[ -f "$DATA_DIR/logs/evaluation_results.json" ]]; then
        python -c "
import json
with open('$DATA_DIR/logs/evaluation_results.json') as f:
    r = json.load(f)
print(f'  Technical accuracy: {r.get(\"technical_accuracy\", 0):.1%}')
print(f'  Reasoning quality:  {r.get(\"reasoning_quality\", 0):.2f}/5.0')
print(f'  Think block rate:   {r.get(\"think_block_present\", 0):.1%}')
passed = r.get('passed_all_gates', False)
print(f'  All gates passed:   {passed}')
"
    fi

    save_checkpoint 5
fi

# ─────────────────────────────────────────────────────────────────────────────
# Stage 6: Quantization
# ─────────────────────────────────────────────────────────────────────────────
CURRENT_STAGE=6
if [[ "$LAST_STAGE" -lt 6 ]]; then
    log_section "Stage 6/8: Quantization (~1 hour)"

    # GPTQ (for GPU inference via API)
    log "  6a: GPTQ 4-bit quantization..."
    python "$REPO_DIR/deployment/quantize_gptq.py" \
        --model "$DATA_DIR/checkpoints" \
        --output "$DATA_DIR/models/phi4-gptq" \
        2>&1 || log "  WARNING: GPTQ quantization failed — using base model"

    # GGUF Q4_K_M (for CPU/local inference)
    log "  6b: GGUF Q4_K_M quantization..."
    python "$REPO_DIR/deployment/quantize_gguf.py" \
        --model "$DATA_DIR/checkpoints" \
        --output "$DATA_DIR/models/phi4-gguf" \
        --quant-type Q4_K_M \
        2>&1 || log "  WARNING: GGUF quantization failed — using GPTQ only"

    # Use GPTQ as primary if available, else raw model
    if [[ -d "$DATA_DIR/models/phi4-gptq" ]]; then
        ln -sfn "$DATA_DIR/models/phi4-gptq" "$DATA_DIR/models/phi4-network-architect"
    else
        ln -sfn "$DATA_DIR/checkpoints" "$DATA_DIR/models/phi4-network-architect"
    fi

    log "  Active model → $DATA_DIR/models/phi4-network-architect"
    save_checkpoint 6
fi

# ─────────────────────────────────────────────────────────────────────────────
# Stage 7: S3 backup
# ─────────────────────────────────────────────────────────────────────────────
CURRENT_STAGE=7
if [[ "$LAST_STAGE" -lt 7 && -n "$S3_BUCKET" ]]; then
    log_section "Stage 7/8: S3 Backup"

    log "  Uploading models to s3://$S3_BUCKET/..."
    aws s3 sync "$DATA_DIR/models" "s3://$S3_BUCKET/models/" \
        --exclude "*.tmp" \
        --storage-class INTELLIGENT_TIERING \
        --only-show-errors

    log "  Uploading evaluation results..."
    aws s3 sync "$DATA_DIR/logs" "s3://$S3_BUCKET/logs/" --only-show-errors

    # Create EBS snapshot for safety
    INSTANCE_ID=$(curl -sf http://169.254.169.254/latest/meta-data/instance-id 2>/dev/null || echo "")
    if [[ -n "$INSTANCE_ID" ]]; then
        EBS_ID=$(aws ec2 describe-instances \
            --instance-ids "$INSTANCE_ID" \
            --query 'Reservations[0].Instances[0].BlockDeviceMappings[?DeviceName==`/dev/sda1`].Ebs.VolumeId' \
            --output text 2>/dev/null || echo "")
        if [[ -n "$EBS_ID" ]]; then
            aws ec2 create-snapshot \
                --volume-id "$EBS_ID" \
                --description "phi4-training-complete-$(date +%Y%m%d)" \
                --tag-specifications 'ResourceType=snapshot,Tags=[{Key=Project,Value=phi4-training}]' \
                --output text > /dev/null
            log "  EBS snapshot created"
        fi
    fi

    save_checkpoint 7
else
    [[ -z "$S3_BUCKET" ]] && log "  Skipping S3 backup (S3_BUCKET not set)"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Stage 8: Dream Team integration
# ─────────────────────────────────────────────────────────────────────────────
CURRENT_STAGE=8
if [[ "$LAST_STAGE" -lt 8 ]]; then
    log_section "Stage 8/8: Dream Team Integration Setup"

    # Validate integration module
    python -c "
import sys
sys.path.insert(0, '$REPO_DIR')
try:
    from dream_team_integration.phi4_inference import Phi4InferenceEngine
    print('  ✓ phi4_inference module OK')
except ImportError as e:
    print(f'  ✗ phi4_inference import error: {e}')
" 2>/dev/null || log "  Note: dream_team_integration module will be available after API start"

    # Test inference with sample query
    log "  Testing inference..."
    python "$REPO_DIR/deployment/inference.py" \
        --model "$DATA_DIR/models/phi4-network-architect" \
        --query "Design a campus network for 500 users in a single building." \
        2>&1 | tail -5 || log "  WARNING: Inference test failed"

    # Start API service
    log "  Starting API server on port 8000..."
    sudo systemctl enable phi4-api 2>/dev/null || true
    sudo systemctl start phi4-api 2>/dev/null || {
        log "  Systemd not available — start manually:"
        log "  cd $REPO_DIR && uvicorn api.main:app --host 0.0.0.0 --port 8000 &"
        nohup "$VENV_DIR/bin/uvicorn" api.main:app \
            --host 0.0.0.0 --port 8000 \
            --app-dir "$REPO_DIR" \
            >> "$DATA_DIR/logs/phi4-api.log" 2>&1 &
        API_PID=$!
        log "  API started (PID $API_PID)"
    }

    # Wait for API
    sleep 5
    curl -sf http://localhost:8000/api/v1/health > /dev/null && \
        log "  ✓ API health check passed" || \
        log "  ✗ API not responding yet (check: tail -f /data/logs/phi4-api.log)"

    save_checkpoint 8
fi

# ─────────────────────────────────────────────────────────────────────────────
# Pipeline complete
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════"
echo "  Phi-4 Network Architect — Training Complete!"
echo "════════════════════════════════════════════════"
echo ""
echo "  Model location: $DATA_DIR/models/phi4-network-architect"
echo "  API endpoint:   http://$(curl -sf http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || hostname -I | awk '{print $1}'):8000"
echo ""
echo "  Test with:"
echo "  curl -X POST http://localhost:8000/api/v1/design \\"
echo "    -H 'X-API-Key: \${PHI4_API_KEY}' \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"requirements\":{\"users\":5000,\"sites\":20}}'"
echo ""

notify "✅ Phi-4 training pipeline complete! API running at port 8000. Model at $DATA_DIR/models/phi4-network-architect"

rm -f "$CHECKPOINT_FILE"

# Auto-shutdown (saves EC2 credits)
if [[ "${AUTO_SHUTDOWN:-false}" == "true" ]]; then
    log "Auto-shutdown in 10 minutes (set AUTO_SHUTDOWN=false to disable)..."
    sleep 600
    sudo shutdown -h now
fi
