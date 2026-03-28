#!/usr/bin/env python3
"""
train_agentic.py — Stage 3: Agentic SFT with custom token-weight loss.

Token weights:
  <think>...</think>          → 2.0 (reasoning is core)
  <tool_call>...</tool_call>  → 1.5 (tool selection critical)
  <|tool_response|>...<|end|> → 0.0 (model didn't generate these)
  Everything else              → 1.0

The tool_response masking is CRITICAL — without it the model learns to
hallucinate tool responses instead of calling tools.
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
from transformers import TrainerCallback
from trl import SFTTrainer, SFTConfig
from unsloth import FastLanguageModel

warnings.filterwarnings("ignore", category=UserWarning)


class AgenticTokenWeightBuilder:
    """Builds per-token loss weights using regex on text regions."""

    def __init__(self, tokenizer, think_weight=2.0, tool_call_weight=1.5,
                 tool_response_weight=0.0, default_weight=1.0):
        self.tokenizer = tokenizer
        self.weights = {
            "think": think_weight,
            "tool_call": tool_call_weight,
            "tool_response": tool_response_weight,
            "default": default_weight,
        }

    def build_weights_from_messages(self, messages: list) -> list:
        """Build per-token weights; user/system/tool turns get 0.0 weight."""
        all_weights = []
        for message in messages:
            role = message["role"]
            content = message["content"]
            tokens = self.tokenizer.encode(content, add_special_tokens=False)

            if role in ("system", "user", "tool"):
                all_weights.extend([0.0] * len(tokens))
            else:
                # Assistant turn — apply region-based weights
                weights = self._region_weights(content, len(tokens))
                all_weights.extend(weights)
        return all_weights

    def _region_weights(self, text: str, n_tokens: int) -> list:
        weights = [self.weights["default"]] * n_tokens
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) != n_tokens:
            return weights

        # Decode each token to find character positions
        decoded = [self.tokenizer.decode([t]) for t in tokens]
        cumlen = [0]
        for d in decoded:
            cumlen.append(cumlen[-1] + len(d))

        regions = []
        for m in re.finditer(r"<think>.*?</think>", text, re.DOTALL):
            regions.append((m.start(), m.end(), self.weights["think"]))
        for m in re.finditer(r"<tool_call>.*?</tool_call>", text, re.DOTALL):
            regions.append((m.start(), m.end(), self.weights["tool_call"]))
        for m in re.finditer(r"<\|tool_response\|>.*?<\|end\|>", text, re.DOTALL):
            regions.append((m.start(), m.end(), self.weights["tool_response"]))

        for char_start, char_end, w in regions:
            for i in range(n_tokens):
                if cumlen[i] >= char_start and cumlen[i] < char_end:
                    weights[i] = w
        return weights


class AgenticSFTTrainer(SFTTrainer):
    """SFTTrainer with custom per-token loss weights."""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        token_weights = inputs.pop("token_weights", None)
        outputs = model(**inputs)
        logits = outputs.logits

        labels = inputs.get("labels", inputs["input_ids"][..., 1:].contiguous())
        if logits.shape[1] > labels.shape[1]:
            logits = logits[..., :labels.shape[1], :]
        elif labels.shape[1] > logits.shape[1]:
            labels = labels[..., :logits.shape[1]]

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        b, seq, v = logits.shape
        loss_per_token = loss_fct(logits.reshape(-1, v), labels.reshape(-1)).view(b, seq)

        mask = (labels != -100).float()
        if token_weights is not None:
            tw = token_weights[..., :seq] if token_weights.shape[-1] >= seq else \
                torch.cat([token_weights, torch.ones(b, seq - token_weights.shape[-1],
                           device=token_weights.device)], dim=-1)
            loss = (loss_per_token * mask * tw).sum() / (mask * tw).sum().clamp(min=1e-8)
        else:
            loss = (loss_per_token * mask).sum() / mask.sum().clamp(min=1e-8)

        return (loss, outputs) if return_outputs else loss


def load_agentic_dataset(data_path: str, tokenizer, config: dict) -> Dataset:
    tw_cfg = config["token_weights"]
    builder = AgenticTokenWeightBuilder(
        tokenizer,
        think_weight=tw_cfg["think_tokens"],
        tool_call_weight=tw_cfg["tool_call_tokens"],
        tool_response_weight=tw_cfg["tool_response_tokens"],
        default_weight=tw_cfg["default_tokens"],
    )

    samples = []
    with open(data_path) as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    formatted = []
    for sample in samples:
        messages = sample["messages"]
        full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        weights = builder.build_weights_from_messages(messages)
        encoded = tokenizer(full_text, truncation=True,
                            max_length=config["data"]["max_seq_length"],
                            return_tensors="pt")
        input_ids = encoded["input_ids"][0].tolist()
        if len(weights) > len(input_ids):
            weights = weights[:len(input_ids)]
        elif len(weights) < len(input_ids):
            weights.extend([1.0] * (len(input_ids) - len(weights)))
        formatted.append({"input_ids": input_ids, "token_weights": weights, "text": full_text})

    return Dataset.from_list(formatted)


def train(config: dict, base_model_path: str, data_path: str, output_dir: str) -> None:
    print(f"\n{'='*60}")
    print("Stage 3: Agentic SFT Training")
    print(f"  think={config['token_weights']['think_tokens']}x  "
          f"tool_call={config['token_weights']['tool_call_tokens']}x  "
          f"tool_response={config['token_weights']['tool_response_tokens']}x (masked)")
    print(f"{'='*60}\n")

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

    dataset = load_agentic_dataset(data_path, tokenizer, config)
    print(f"Loaded {len(dataset)} agentic samples")

    train_cfg = config["training"]
    sft_config = SFTConfig(
        learning_rate=float(train_cfg["learning_rate"]),
        num_train_epochs=train_cfg["num_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        warmup_ratio=train_cfg["warmup_ratio"],
        weight_decay=train_cfg["weight_decay"],
        max_grad_norm=train_cfg["max_grad_norm"],
        output_dir=output_dir,
        save_steps=train_cfg["save_steps"],
        logging_steps=train_cfg["logging_steps"],
        seed=train_cfg["seed"],
        fp16=False, bf16=True,
        report_to="wandb" if os.environ.get("WANDB_API_KEY") else "none",
        max_seq_length=config["data"]["max_seq_length"],
        dataset_text_field="text",
        remove_unused_columns=False,
    )

    trainer = AgenticSFTTrainer(model=model, tokenizer=tokenizer,
                                args=sft_config, train_dataset=dataset)
    trainer.train()

    adapter_path = Path(output_dir) / "adapter"
    adapter_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print(f"\n✓ Agentic SFT complete — adapter: {adapter_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_agentic.yaml")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--data-path", default="/data/datasets/agentic_train.jsonl")
    parser.add_argument("--output-dir", default="/data/stage3")
    args = parser.parse_args()

    with open(Path(__file__).parent / args.config) as f:
        config = yaml.safe_load(f)
    config["model"]["name"] = args.base_model
    train(config, args.base_model, args.data_path, args.output_dir)


if __name__ == "__main__":
    main()
