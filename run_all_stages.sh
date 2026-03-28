#!/bin/bash
# run_all_stages.sh — 3-stage Phi-4 fine-tuning orchestrator
# Usage: bash run_all_stages.sh [--stage 1|2|3] [--resume]
set -euo pipefail

BASE_DIR="/data"
STATE_FILE="$BASE_DIR/pipeline_state.json"
LOG_DIR="$BASE_DIR/logs"
REPO_DIR="/opt/phi4"
mkdir -p "$LOG_DIR"

# ── Argument parsing ─────────────────────────────────────────────────────────
START_STAGE=0   # 0 = start from dataset generation
RESUME=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage) START_STAGE="$2"; shift 2 ;;
    --resume) RESUME=true; shift ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

# ── State management ─────────────────────────────────────────────────────────
init_state() {
  if [[ ! -f "$STATE_FILE" ]] || [[ "$RESUME" == "false" ]]; then
    cat > "$STATE_FILE" <<EOF
{
  "pipeline_version": "1.0",
  "started_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "stages": {
    "1": {"status": "pending", "started_at": null, "completed_at": null},
    "2": {"status": "pending", "started_at": null, "completed_at": null},
    "3": {"status": "pending", "started_at": null, "completed_at": null}
  },
  "merge_completed": {"1": false, "2": false, "3": false}
}
EOF
    echo "[STATE] Initialized fresh pipeline state"
  else
    echo "[STATE] Resuming from existing state"
    cat "$STATE_FILE"
  fi
}

update_state() {
  local stage="$1" key="$2" value="$3"
  python3 -c "
import json
with open('$STATE_FILE') as f: s = json.load(f)
s['stages']['$stage']['$key'] = '$value'
with open('$STATE_FILE', 'w') as f: json.dump(s, f, indent=2)
"
}

mark_merge_done() {
  python3 -c "
import json
with open('$STATE_FILE') as f: s = json.load(f)
s['merge_completed']['$1'] = True
with open('$STATE_FILE', 'w') as f: json.dump(s, f, indent=2)
"
}

# ── SIGTERM handler ───────────────────────────────────────────────────────────
trap 'echo "[SIGNAL] SIGTERM received — saving state and exiting gracefully"; exit 0' SIGTERM SIGINT

# ── Dataset generation ────────────────────────────────────────────────────────
run_dataset_generation() {
  echo "════════════════════════════════════════"
  echo "[DATA] Starting dataset generation..."
  echo "════════════════════════════════════════"

  cd "$REPO_DIR/data_generation"

  if [[ -f "$BASE_DIR/datasets/sft_train.jsonl" ]] && [[ "$RESUME" == "true" ]]; then
    echo "[DATA] SFT dataset exists, skipping..."
  else
    echo "[DATA] Generating SFT dataset (9,000 samples)..."
    python run_pipeline.py --type sft --output "$BASE_DIR/datasets/" \
      2>&1 | tee "$LOG_DIR/data_sft.log"
  fi

  if [[ -f "$BASE_DIR/datasets/grpo_train.jsonl" ]] && [[ "$RESUME" == "true" ]]; then
    echo "[DATA] GRPO dataset exists, skipping..."
  else
    echo "[DATA] Generating GRPO dataset (800 samples)..."
    python run_pipeline.py --type grpo --output "$BASE_DIR/datasets/" \
      2>&1 | tee "$LOG_DIR/data_grpo.log"
  fi

  if [[ -f "$BASE_DIR/datasets/agentic_train.jsonl" ]] && [[ "$RESUME" == "true" ]]; then
    echo "[DATA] Agentic dataset exists, skipping..."
  else
    echo "[DATA] Generating Agentic dataset (2,000 samples)..."
    python run_pipeline.py --type agentic --output "$BASE_DIR/datasets/" \
      2>&1 | tee "$LOG_DIR/data_agentic.log"
  fi

  echo "[DATA] All datasets ready"
}

# ── LoRA merge helper ─────────────────────────────────────────────────────────
merge_lora() {
  local stage="$1"
  local adapter_path="$BASE_DIR/stage${stage}/adapter"
  local merged_path="$BASE_DIR/merged/stage${stage}"
  local base

  if [[ "$stage" == "1" ]]; then
    base="microsoft/Phi-4"
  else
    base="$BASE_DIR/merged/stage$(($stage - 1))"
  fi

  echo "[MERGE] Merging Stage ${stage} LoRA adapters..."
  python3 - <<PYEOF 2>&1 | tee "$LOG_DIR/merge_stage${stage}.log"
from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="$base",
    max_seq_length=4096,
    dtype=torch.bfloat16,
    load_in_4bit=False,
)
model.load_adapter("$adapter_path")
merged = model.merge_and_unload()
merged.save_pretrained("$merged_path")
tokenizer.save_pretrained("$merged_path")
print("[MERGE] Stage $stage merge complete: $merged_path")
PYEOF

  mark_merge_done "$stage"
  echo "[MERGE] Stage ${stage} → $merged_path"
}

# ── OOM retry wrapper ─────────────────────────────────────────────────────────
run_with_oom_retry() {
  local max_retries=2
  local attempt=0

  while [[ $attempt -lt $max_retries ]]; do
    if "$@"; then
      return 0
    else
      attempt=$((attempt + 1))
      echo "[OOM] Command failed, attempt $attempt/$max_retries — freeing GPU memory..."
      python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
      sleep 30
    fi
  done

  echo "[ERROR] Command failed after $max_retries attempts"
  return 1
}

# ── Stage 1: SFT ─────────────────────────────────────────────────────────────
run_stage1() {
  echo "════════════════════════════════════════"
  echo "[STAGE 1] SFT Training (Phi-4 base → CCDE specialist)"
  echo "════════════════════════════════════════"
  update_state "1" "status" "running"
  update_state "1" "started_at" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"

  cd "$REPO_DIR/fine_tuning"
  run_with_oom_retry python train.py \
    --config config.yaml \
    --data-path "$BASE_DIR/datasets/sft_train.jsonl" \
    --output-dir "$BASE_DIR/stage1" \
    2>&1 | tee "$LOG_DIR/stage1_train.log"

  update_state "1" "status" "completed"
  update_state "1" "completed_at" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"

  echo "[STAGE 1] Evaluating..."
  cd "$REPO_DIR"
  python evaluation/stage_tracker.py --stage 1 --model "$BASE_DIR/stage1/adapter" \
    2>&1 | tee "$LOG_DIR/eval_stage1.log"

  merge_lora 1
  echo "[STAGE 1] Complete ✓"
}

# ── Stage 2: GRPO ─────────────────────────────────────────────────────────────
run_stage2() {
  echo "════════════════════════════════════════"
  echo "[STAGE 2] GRPO Training (reasoning optimization)"
  echo "════════════════════════════════════════"
  update_state "2" "status" "running"
  update_state "2" "started_at" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"

  if [[ ! -d "$BASE_DIR/merged/stage1" ]]; then
    echo "[ERROR] Stage 1 merged model not found at $BASE_DIR/merged/stage1"
    exit 1
  fi

  export VLLM_GPU_MEMORY_UTILIZATION=0.45
  cd "$REPO_DIR/fine_tuning"
  run_with_oom_retry python train_grpo.py \
    --config config_grpo.yaml \
    --base-model "$BASE_DIR/merged/stage1" \
    --data-path "$BASE_DIR/datasets/grpo_train.jsonl" \
    --output-dir "$BASE_DIR/stage2" \
    2>&1 | tee "$LOG_DIR/stage2_grpo.log"

  update_state "2" "status" "completed"
  update_state "2" "completed_at" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"

  echo "[STAGE 2] Evaluating..."
  cd "$REPO_DIR"
  python evaluation/stage_tracker.py --stage 2 --model "$BASE_DIR/stage2/adapter" \
    2>&1 | tee "$LOG_DIR/eval_stage2.log"

  merge_lora 2
  echo "[STAGE 2] Complete ✓"
}

# ── Stage 3: Agentic SFT ─────────────────────────────────────────────────────
run_stage3() {
  echo "════════════════════════════════════════"
  echo "[STAGE 3] Agentic SFT (tool use + ReAct)"
  echo "════════════════════════════════════════"
  update_state "3" "status" "running"
  update_state "3" "started_at" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"

  if [[ ! -d "$BASE_DIR/merged/stage2" ]]; then
    echo "[ERROR] Stage 2 merged model not found at $BASE_DIR/merged/stage2"
    exit 1
  fi

  cd "$REPO_DIR/fine_tuning"
  run_with_oom_retry python train_agentic.py \
    --config config_agentic.yaml \
    --base-model "$BASE_DIR/merged/stage2" \
    --data-path "$BASE_DIR/datasets/agentic_train.jsonl" \
    --output-dir "$BASE_DIR/stage3" \
    2>&1 | tee "$LOG_DIR/stage3_agentic.log"

  update_state "3" "status" "completed"
  update_state "3" "completed_at" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"

  echo "[STAGE 3] Evaluating..."
  cd "$REPO_DIR"
  python evaluation/stage_tracker.py --stage 3 --model "$BASE_DIR/stage3/adapter" \
    2>&1 | tee "$LOG_DIR/eval_stage3.log"

  python evaluation/compare_models.py --output COMPARISON_REPORT.md \
    2>&1 | tee "$LOG_DIR/comparison.log"

  python evaluation/agentic_eval.py \
    --model "$BASE_DIR/merged/stage3" \
    --output "$LOG_DIR/agentic_eval_results.json" \
    2>&1 | tee "$LOG_DIR/agentic_eval.log"

  merge_lora 3
  echo "[STAGE 3] Complete ✓"
}

# ── Main ──────────────────────────────────────────────────────────────────────
echo "╔══════════════════════════════════════════╗"
echo "║  Phi-4 3-Stage Fine-Tuning Pipeline      ║"
echo "║  g5.2xlarge | A10G 24GB | 500GB EBS      ║"
echo "╚══════════════════════════════════════════╝"

init_state

[[ $START_STAGE -le 0 ]] && run_dataset_generation
[[ $START_STAGE -le 1 ]] && run_stage1
[[ $START_STAGE -le 2 ]] && run_stage2
[[ $START_STAGE -le 3 ]] && run_stage3

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║  Pipeline COMPLETE ✓                     ║"
echo "║  Final model: /data/merged/stage3        ║"
echo "╚══════════════════════════════════════════╝"
cat "$STATE_FILE"
