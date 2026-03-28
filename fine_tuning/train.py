"""
Phi-4 14B Fine-Tuning Script
Network Architect with CCDE-Level Reasoning

Uses Unsloth for memory-efficient LoRA training with reasoning-weighted loss.
Phi-4 specific: <think>...</think> tokens get 2x loss weight.

Usage:
    # Standard training
    python fine_tuning/train.py

    # Custom config
    python fine_tuning/train.py --config fine_tuning/config.yaml

    # Dry run (validates setup only)
    python fine_tuning/train.py --dry-run

    # Resume from checkpoint
    python fine_tuning/train.py --resume models/phi4-network-architect/checkpoint-500

Requirements:
    pip install -r fine_tuning/requirements.txt
    # CUDA 12.1+ required
    pip install torch --index-url https://download.pytorch.org/whl/cu121
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


# ─────────────────────────────────────────────────────────────
# Custom reasoning-weighted loss
# ─────────────────────────────────────────────────────────────

class ReasoningWeightedTrainer:
    """
    Wraps TRL's SFTTrainer to apply 2x loss weight on <think> tokens.

    Phi-4 uses <|think|> style tokens; we find them by scanning token ids
    matching the think_start and think_end strings in the tokenizer vocab.
    """

    @staticmethod
    def get_think_token_ids(tokenizer: Any) -> tuple[list[int], list[int]]:
        """Find token IDs for <think> and </think> tags."""
        think_start_ids: list[int] = []
        think_end_ids: list[int] = []

        # Try direct encoding
        for candidate in ["<think>", "<|think|>", "[THINK]"]:
            ids = tokenizer.encode(candidate, add_special_tokens=False)
            if ids:
                think_start_ids = ids
                break

        for candidate in ["</think>", "<|/think|>", "[/THINK]"]:
            ids = tokenizer.encode(candidate, add_special_tokens=False)
            if ids:
                think_end_ids = ids
                break

        log.info(f"Think start token IDs: {think_start_ids}")
        log.info(f"Think end token IDs: {think_end_ids}")
        return think_start_ids, think_end_ids

    @staticmethod
    def build_token_weights(
        input_ids: torch.Tensor,
        think_start_ids: list[int],
        think_end_ids: list[int],
        think_weight: float = 2.0,
    ) -> torch.Tensor:
        """
        Build per-token weight tensor.
        Tokens inside <think>...</think> get weight=think_weight.
        All other tokens get weight=1.0.

        Args:
            input_ids: Shape (batch, seq_len)
            think_start_ids: Token IDs for <think>
            think_end_ids: Token IDs for </think>
            think_weight: Multiplier for think tokens

        Returns:
            weights tensor of shape (batch, seq_len)
        """
        weights = torch.ones_like(input_ids, dtype=torch.float32)

        if not think_start_ids or not think_end_ids:
            return weights

        for batch_idx in range(input_ids.shape[0]):
            ids = input_ids[batch_idx].tolist()
            in_think = False
            i = 0
            while i < len(ids):
                # Check for think start
                if ids[i:i + len(think_start_ids)] == think_start_ids:
                    in_think = True
                    for j in range(len(think_start_ids)):
                        weights[batch_idx, i + j] = think_weight
                    i += len(think_start_ids)
                    continue
                # Check for think end
                if ids[i:i + len(think_end_ids)] == think_end_ids:
                    for j in range(len(think_end_ids)):
                        weights[batch_idx, i + j] = think_weight
                    in_think = False
                    i += len(think_end_ids)
                    continue
                # Apply weight if inside think block
                if in_think:
                    weights[batch_idx, i] = think_weight
                i += 1

        return weights


# ─────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────

def load_jsonl_dataset(path: Path, tokenizer: Any, max_seq_length: int) -> Any:
    """Load JSONL dataset and apply Phi-4 chat template."""
    from datasets import Dataset

    samples: list[dict[str, str]] = []
    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                log.warning(f"Skipping malformed JSON at line {line_num}")
                continue

            # Extract messages (standard OpenAI format)
            messages = item.get("messages", [])
            if not messages:
                continue

            # Apply Phi-4 chat template
            try:
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            except Exception:
                # Fallback: manual formatting
                text = _manual_phi4_format(messages)

            samples.append({"text": text})

    log.info(f"Loaded {len(samples)} samples from {path}")
    return Dataset.from_list(samples)


def _manual_phi4_format(messages: list[dict]) -> str:
    """Manual Phi-4 chat template as fallback."""
    parts = []
    system_msg = next((m for m in messages if m["role"] == "system"), None)
    if system_msg:
        parts.append(f"<|system|>\n{system_msg['content']}<|end|>")

    for msg in messages:
        if msg["role"] == "system":
            continue
        role = "user" if msg["role"] == "user" else "assistant"
        parts.append(f"<|{role}|>\n{msg['content']}<|end|>")

    if messages and messages[-1]["role"] == "user":
        parts.append("<|assistant|>")

    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────

def train(config: dict[str, Any], resume_from: str | None = None) -> None:
    """Main training function."""
    from unsloth import FastLanguageModel  # type: ignore[import]
    # SFTTrainer and SFTConfig are exported at the trl package level at runtime
    from trl import SFTTrainer, SFTConfig  # type: ignore[import]

    model_cfg = config["model"]
    lora_cfg = config["lora"]
    train_cfg = config["training"]
    data_cfg = config["data"]
    out_cfg = config["output"]
    reasoning_cfg = config.get("reasoning_loss", {})
    wandb_cfg = config.get("wandb", {})

    # ── Configure W&B ────────────────────────────────────────
    if wandb_cfg:
        os.environ.setdefault("WANDB_PROJECT", wandb_cfg.get("project", "phi4-network-architect"))
        run_name = wandb_cfg.get("run_name", "phi4-training")
    else:
        run_name = "phi4-training"

    # ── Load base model ──────────────────────────────────────
    log.info(f"Loading base model: {model_cfg['name']}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_cfg["name"],
        max_seq_length=model_cfg["max_seq_length"],
        load_in_4bit=model_cfg["load_in_4bit"],
        dtype=None,  # Auto-detect
    )
    log.info(f"Model loaded. Params: {model.num_parameters():,}")

    # ── Apply LoRA ───────────────────────────────────────────
    log.info(f"Applying LoRA (r={lora_cfg['r']}, alpha={lora_cfg['lora_alpha']})")
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_cfg["r"],
        target_modules=lora_cfg["target_modules"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
        use_gradient_checkpointing=lora_cfg["use_gradient_checkpointing"],
        random_state=lora_cfg["random_state"],
        use_rslora=lora_cfg["use_rslora"],
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log.info(f"LoRA params: {trainable:,} / {total:,} ({100*trainable/total:.2f}% trainable)")

    # ── Load datasets ────────────────────────────────────────
    train_path = ROOT / data_cfg["train_file"]
    val_path = ROOT / data_cfg["val_file"]

    if not train_path.exists():
        log.error(f"Train data not found: {train_path}")
        log.error("Run: python data_generation/dataset_generator.py --count 10000")
        sys.exit(1)

    train_dataset = load_jsonl_dataset(train_path, tokenizer, model_cfg["max_seq_length"])
    val_dataset = load_jsonl_dataset(val_path, tokenizer, model_cfg["max_seq_length"]) if val_path.exists() else None

    if data_cfg.get("max_samples"):
        train_dataset = train_dataset.select(range(min(data_cfg["max_samples"], len(train_dataset))))

    log.info(f"Train samples: {len(train_dataset)}")
    if val_dataset:
        log.info(f"Val samples: {len(val_dataset)}")

    # ── Configure trainer ────────────────────────────────────
    output_dir = ROOT / out_cfg["dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    sft_config = SFTConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        warmup_steps=train_cfg["warmup_steps"],
        num_train_epochs=train_cfg["num_train_epochs"],
        learning_rate=train_cfg["learning_rate"],
        fp16=train_cfg.get("fp16", False),
        bf16=train_cfg.get("bf16", True),
        logging_steps=train_cfg["logging_steps"],
        eval_steps=train_cfg.get("eval_steps", 250),
        save_steps=train_cfg["save_steps"],
        save_total_limit=train_cfg["save_total_limit"],
        optim=train_cfg["optim"],
        weight_decay=train_cfg["weight_decay"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        seed=train_cfg["seed"],
        eval_strategy="steps" if val_dataset else "no",
        load_best_model_at_end=train_cfg.get("load_best_model_at_end", True) if val_dataset else False,
        report_to=["wandb"],
        run_name=run_name,
        # TRL 0.23+: dataset params moved from SFTTrainer to SFTConfig
        dataset_text_field="text",
        max_length=model_cfg["max_seq_length"],
        packing=False,
    )

    # ── Build trainer (with optional reasoning-weighted loss) ──
    use_weighted_loss = reasoning_cfg.get("enabled", False)
    think_start_ids: list[int] = []
    think_end_ids: list[int] = []
    think_weight_val = reasoning_cfg.get("think_token_weight", 2.0)

    if use_weighted_loss:
        think_start_ids, think_end_ids = ReasoningWeightedTrainer.get_think_token_ids(tokenizer)
        if think_start_ids and think_end_ids:
            log.info(f"Reasoning-weighted loss enabled (think_weight={think_weight_val})")
        else:
            log.warning("Could not find <think> tokens - using standard loss")
            use_weighted_loss = False

    if use_weighted_loss:
        _s_ids, _e_ids, _w = think_start_ids, think_end_ids, think_weight_val

        class WeightedSFTTrainer(SFTTrainer):  # type: ignore[misc]
            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):  # type: ignore[override]
                iids = inputs.get("input_ids")
                if iids is not None:
                    inputs = dict(inputs)
                    inputs["weights"] = ReasoningWeightedTrainer.build_token_weights(
                        iids, _s_ids, _e_ids, _w
                    ).to(model.device)
                return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

        TrainerCls = WeightedSFTTrainer
    else:
        TrainerCls = SFTTrainer

    trainer = TrainerCls(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=sft_config,
    )

    # ── Train ────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("Starting training")
    log.info(f"  Model: {model_cfg['name']}")
    log.info(f"  Train samples: {len(train_dataset)}")
    log.info(f"  Epochs: {train_cfg['num_train_epochs']}")
    log.info(f"  Effective batch size: {train_cfg['per_device_train_batch_size'] * train_cfg['gradient_accumulation_steps']}")
    log.info(f"  Output: {output_dir}")
    log.info("=" * 60)

    trainer_stats = trainer.train(resume_from_checkpoint=resume_from)

    # ── Save final model ─────────────────────────────────────
    log.info("Saving final model...")
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Save LoRA adapters separately
    lora_dir = output_dir / "lora_adapters"
    lora_dir.mkdir(exist_ok=True)
    model.save_pretrained(str(lora_dir))
    log.info(f"LoRA adapters saved to: {lora_dir}")

    # ── Print stats ──────────────────────────────────────────
    log.info("=" * 60)
    log.info("Training complete!")
    log.info(f"  Training time: {trainer_stats.metrics.get('train_runtime', 0):.0f}s")
    log.info(f"  Samples/sec: {trainer_stats.metrics.get('train_samples_per_second', 0):.2f}")
    log.info(f"  Final train loss: {trainer_stats.metrics.get('train_loss', 0):.4f}")
    log.info(f"  Peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
    log.info(f"  Model saved to: {output_dir}")
    log.info("=" * 60)

    # ── Push to Hub (optional) ───────────────────────────────
    hub_model_id = out_cfg.get("hub_model_id")
    hub_token = out_cfg.get("hub_token") or os.environ.get("HF_TOKEN")
    if hub_model_id and hub_token:
        log.info(f"Pushing to HuggingFace Hub: {hub_model_id}")
        model.push_to_hub(hub_model_id, token=hub_token)
        tokenizer.push_to_hub(hub_model_id, token=hub_token)
        log.info("Upload complete!")


# ─────────────────────────────────────────────────────────────
# Validation / dry-run
# ─────────────────────────────────────────────────────────────

def validate_setup(config: dict[str, Any]) -> bool:
    """Validate environment and data before full training run."""
    ok = True

    log.info("Validating setup...")

    # Check CUDA
    if not torch.cuda.is_available():
        log.error("CUDA not available - training will be very slow (CPU only)")
        ok = False
    else:
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        log.info(f"GPU: {torch.cuda.get_device_name(0)} ({vram_gb:.1f} GB)")
        if vram_gb < 20:
            log.warning(f"Only {vram_gb:.1f} GB VRAM - recommend 24+ GB for Phi-4 14B")

    # Check Unsloth
    try:
        import unsloth  # noqa: F401
        log.info("Unsloth: OK")
    except ImportError:
        log.error("Unsloth not installed: pip install unsloth")
        ok = False

    # Check train data
    train_path = ROOT / config["data"]["train_file"]
    if not train_path.exists():
        log.error(f"Train data missing: {train_path}")
        log.error("Generate first: python data_generation/dataset_generator.py --count 10000")
        ok = False
    else:
        count = sum(1 for _ in open(train_path))
        log.info(f"Train samples: {count} ({train_path})")
        if count < 1000:
            log.warning(f"Only {count} training samples - recommend 10,000+")

    # Check output dir
    output_dir = ROOT / config["output"]["dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Output dir: {output_dir}")

    return ok


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict[str, Any]:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phi-4 Network Architect Fine-Tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parent / "config.yaml"),
        help="Path to training config YAML",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate setup without training",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Resume from checkpoint path",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit training samples (for quick tests)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.max_samples:
        config["data"]["max_samples"] = args.max_samples

    if args.dry_run:
        ok = validate_setup(config)
        sys.exit(0 if ok else 1)

    if not validate_setup(config):
        log.error("Setup validation failed. Fix issues before training.")
        sys.exit(1)

    train(config, resume_from=args.resume)


if __name__ == "__main__":
    main()
