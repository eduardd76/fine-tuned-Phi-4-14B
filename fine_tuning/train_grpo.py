#!/usr/bin/env python3
"""
train_grpo.py — Stage 2: GRPO training for genuine reasoning capabilities.

GRPO (Group Relative Policy Optimization) teaches the model to REASON toward
correct answers rather than just imitating patterns from SFT.

Key insight: 8 rollouts generated per prompt, scored by reward functions,
model learns to produce rollouts that score like the best ones.

CRITICAL implementation notes:
1. Reward functions must be module-level (not class methods) for TRL serialization
2. kwargs["expected_key_facts"] is a list-of-lists: outer=per-completion, inner=facts
3. VLLM_GPU_MEMORY_UTILIZATION=0.45 MUST be set before this script runs
4. GRPOTrainer needs merged base model, not just LoRA adapters
"""
from __future__ import annotations
import argparse
import json
import os
import re
import warnings
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from trl import GRPOTrainer, GRPOConfig
from unsloth import FastLanguageModel

warnings.filterwarnings("ignore", category=UserWarning)


# ── GRPO Reward Functions ─────────────────────────────────────────────────────
# MUST be module-level for TRL to pickle them for multiprocess rollout evaluation.
# kwargs keys mirror dataset column names — passed as batched lists.
# Total possible reward = 0.50 + 0.25 + 0.15 + 0.10 = 1.00

def reward_keyword_match(prompts: list, completions: list, **kwargs) -> list:
    """
    Reward 0.50 × (fraction of required keywords present).
    CRITICAL: kwargs["expected_key_facts"] is list-of-lists.
    """
    key_facts_batch = kwargs.get("expected_key_facts", [[] for _ in completions])
    scores = []
    for completion, key_facts in zip(completions, key_facts_batch):
        if not key_facts:
            scores.append(0.0)
            continue
        hits = sum(1 for term in key_facts if term.lower() in completion.lower())
        scores.append(0.50 * (hits / len(key_facts)))
    return scores


def reward_exact_values(prompts: list, completions: list, **kwargs) -> list:
    """
    Reward 0.25 × (fraction of exact numerical values present).
    For calculations: checks "52.6", "2048", "3160" are present verbatim.
    """
    exact_batch = kwargs.get("exact_values", [[] for _ in completions])
    scores = []
    for completion, exact_vals in zip(completions, exact_batch):
        if not exact_vals:
            scores.append(0.125)  # Partial credit if no exact values required
            continue
        hits = sum(1 for val in exact_vals if val in completion)
        scores.append(0.25 * (hits / len(exact_vals)))
    return scores


def reward_think_block(prompts: list, completions: list, **kwargs) -> list:
    """
    Reward 0.15 for having a substantive <think> block (≥30 words).
    """
    scores = []
    for completion in completions:
        match = re.search(r"<think>(.*?)</think>", completion, re.DOTALL)
        if match:
            think_words = len(match.group(1).split())
            if think_words >= 50:
                scores.append(0.15)
            elif think_words >= 30:
                scores.append(0.075)
            else:
                scores.append(0.0)
        else:
            scores.append(0.0)
    return scores


def reward_answer_quality(prompts: list, completions: list, **kwargs) -> list:
    """
    Reward 0.10 for answer quality: length, recommendation, no uncertainty.
    """
    scores = []
    uncertainty = [r"\bI'm not sure\b", r"\bI cannot\b", r"\bAs an AI\b", r"\bI don't know\b"]
    recommendation = [r"\brecommend\b", r"\bshould use\b", r"\bdesign\b", r"\bimplement\b", r"\bconfigure\b"]

    for completion in completions:
        score = 0.0
        answer = re.sub(r"<think>.*?</think>", "", completion, flags=re.DOTALL)
        word_count = len(answer.split())

        if word_count >= 100:
            score += 0.05
        elif word_count >= 80:
            score += 0.025

        if any(re.search(p, completion, re.IGNORECASE) for p in recommendation):
            score += 0.03

        if not any(re.search(p, completion, re.IGNORECASE) for p in uncertainty):
            score += 0.02

        scores.append(min(score, 0.10))
    return scores


REWARD_FUNCTIONS = [
    reward_keyword_match,
    reward_exact_values,
    reward_think_block,
    reward_answer_quality,
]


def load_grpo_dataset(data_path: str, tokenizer) -> Dataset:
    """Load and format GRPO dataset — extract prompt only (no response)."""
    samples = []
    with open(data_path) as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    def format_prompt(sample):
        messages = sample["messages"]
        # Take only system + user — model generates the assistant response
        prompt_messages = [m for m in messages if m["role"] != "assistant"]
        prompt = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return {
            "prompt": prompt,
            "expected_key_facts": sample.get("expected_key_facts", []),
            "exact_values": sample.get("exact_values", []),
            "answer_type": sample.get("answer_type", ""),
            "answer_key": sample.get("answer_key", ""),
        }

    return Dataset.from_list([format_prompt(s) for s in samples])


def train(config: dict, base_model_path: str, data_path: str, output_dir: str) -> None:
    """Run GRPO training."""
    print(f"\n{'='*60}")
    print("Stage 2: GRPO Training — Genuine Reasoning Optimization")
    print(f"  Base model: {base_model_path}")
    print(f"  Data: {data_path}")
    print(f"  KL coefficient: {config['grpo']['kl_coeff']}")
    print(f"  Rollouts per prompt: {config['grpo']['num_generations']}")
    print(f"{'='*60}\n")

    # Check VLLM env var
    vllm_util = float(os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.9"))
    if vllm_util > 0.5:
        print(f"WARNING: VLLM_GPU_MEMORY_UTILIZATION={vllm_util} — setting to 0.45 to prevent OOM")
        os.environ["VLLM_GPU_MEMORY_UTILIZATION"] = "0.45"

    print("Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_path,
        max_seq_length=config["model"]["max_seq_length"],
        dtype=torch.bfloat16,
        load_in_4bit=config["model"]["load_in_4bit"],
    )

    lora_cfg = config["lora"]
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_cfg["r"],
        target_modules=lora_cfg["target_modules"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
        use_gradient_checkpointing="unsloth",
        random_state=config["training"]["seed"],
    )

    print("Loading GRPO dataset...")
    dataset = load_grpo_dataset(data_path, tokenizer)
    print(f"  {len(dataset)} samples")

    grpo_cfg = config["grpo"]
    train_cfg = config["training"]

    grpo_config = GRPOConfig(
        num_generations=grpo_cfg["num_generations"],
        temperature=grpo_cfg["temperature"],
        top_p=grpo_cfg["top_p"],
        max_new_tokens=grpo_cfg["max_new_tokens"],
        kl_coeff=grpo_cfg["kl_coeff"],
        learning_rate=float(train_cfg["learning_rate"]),
        num_train_epochs=train_cfg["num_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        warmup_steps=train_cfg["warmup_steps"],
        weight_decay=train_cfg["weight_decay"],
        max_grad_norm=train_cfg["max_grad_norm"],
        output_dir=output_dir,
        save_steps=train_cfg["save_steps"],
        logging_steps=train_cfg["logging_steps"],
        seed=train_cfg["seed"],
        fp16=False,
        bf16=True,
        report_to="wandb" if os.environ.get("WANDB_API_KEY") else "none",
    )

    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        config=grpo_config,
        train_dataset=dataset,
        reward_funcs=REWARD_FUNCTIONS,
    )

    print("Starting GRPO training...")
    trainer.train()

    adapter_path = Path(output_dir) / "adapter"
    adapter_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))

    print(f"\n✓ GRPO training complete — adapter saved: {adapter_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_grpo.yaml")
    parser.add_argument("--base-model", required=True, help="Path to Stage 1 merged model")
    parser.add_argument("--data-path", default="/data/datasets/grpo_train.jsonl")
    parser.add_argument("--output-dir", default="/data/stage2")
    args = parser.parse_args()

    config_path = Path(__file__).parent / args.config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    config["model"]["name"] = args.base_model
    train(config=config, base_model_path=args.base_model,
          data_path=args.data_path, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
