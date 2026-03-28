"""
GPTQ Quantization Script
Quantizes fine-tuned Phi-4 to 4-bit GPTQ format for fast GPU inference.

GPTQ provides:
- 4-bit quantization with minimal accuracy loss
- Faster inference than 4-bit BnB (uses optimized GPU kernels)
- ~8 GB model size (vs ~26 GB float16)
- Compatible with AutoGPTQ, ExLlama2, vLLM

Usage:
    python deployment/quantize_gptq.py \
        --model models/phi4-network-architect \
        --output models/phi4-gptq \
        --bits 4

    # With calibration data from training set
    python deployment/quantize_gptq.py \
        --model models/phi4-network-architect \
        --output models/phi4-gptq \
        --calibration-data generated_data/training_data.jsonl \
        --calibration-samples 128
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def load_calibration_data(
    data_path: Path,
    tokenizer: Any,
    num_samples: int = 128,
    max_length: int = 2048,
) -> list[Any]:
    """Load and tokenize calibration samples for GPTQ."""
    samples: list[str] = []

    with open(data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            messages = item.get("messages", [])
            if messages:
                try:
                    text = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=False
                    )
                except Exception:
                    text = " ".join(m.get("content", "") for m in messages)
                samples.append(text)

    # Random sample
    random.seed(42)
    random.shuffle(samples)
    samples = samples[:num_samples]

    # Tokenize
    tokenized = []
    for text in samples:
        enc = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        tokenized.append(enc["input_ids"])

    print(f"Calibration data: {len(tokenized)} samples loaded")
    return tokenized


def quantize_gptq(
    model_path: Path,
    output_path: Path,
    bits: int = 4,
    group_size: int = 128,
    calibration_data_path: Path | None = None,
    calibration_samples: int = 128,
) -> bool:
    """
    Quantize model to GPTQ format.

    Args:
        model_path: Path to fine-tuned model (merged, no LoRA)
        output_path: Output directory for GPTQ model
        bits: Quantization bits (4 recommended)
        group_size: Group size for quantization (128 recommended)
        calibration_data_path: Path to training JSONL for calibration
        calibration_samples: Number of calibration samples

    Returns:
        True on success
    """
    try:
        from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig  # type: ignore[import]
        from transformers import AutoTokenizer
        import torch
    except ImportError:
        print("auto-gptq not installed: pip install auto-gptq optimum")
        return False

    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return False

    print(f"Loading model for GPTQ quantization...")
    print(f"  Input: {model_path}")
    print(f"  Output: {output_path}")
    print(f"  Bits: {bits}, Group size: {group_size}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load calibration data
    if calibration_data_path and calibration_data_path.exists():
        calibration_data = load_calibration_data(
            calibration_data_path, tokenizer,
            num_samples=calibration_samples,
        )
    else:
        # Use default network architect prompts as calibration
        print("Using built-in calibration data...")
        default_prompts = [
            "Design a network for 5000 users across 20 sites with PCI-DSS compliance.",
            "Troubleshoot BGP neighbor down issue. The neighbor is stuck in Active state.",
            "What are the design criteria for spine-leaf vs three-tier architecture?",
            "Design WAN connectivity for a global enterprise with 50 remote sites.",
            "Explain HSRP vs VRRP vs GLBP selection criteria for enterprise campus.",
            "How do you achieve 99.99% uptime in a data center network design?",
            "Design MPLS L3VPN for a service provider with 100 enterprise customers.",
            "Troubleshoot OSPF neighbor stuck in EXSTART/EXCHANGE state.",
            "What QoS policy is needed for unified communications with voice and video?",
            "Design security segmentation for a healthcare network with HIPAA compliance.",
        ] * 13  # 130 samples

        calibration_data = []
        for prompt in default_prompts[:calibration_samples]:
            enc = tokenizer(
                f"<|user|>\n{prompt}<|end|>\n<|assistant|>",
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            calibration_data.append(enc["input_ids"])

    # GPTQ quantization config
    quantize_config = BaseQuantizeConfig(
        bits=bits,
        group_size=group_size,
        desc_act=True,  # Activation ordering for better accuracy
        sym=True,       # Symmetric quantization
    )

    print("\nLoading model in float16 for quantization...")
    model = AutoGPTQForCausalLM.from_pretrained(
        str(model_path),
        quantize_config=quantize_config,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    print(f"Quantizing with {len(calibration_data)} calibration samples...")
    print("This will take 30-90 minutes depending on hardware...")
    model.quantize(calibration_data, use_triton=False)

    print(f"\nSaving GPTQ model to: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_quantized(str(output_path), use_safetensors=True)
    tokenizer.save_pretrained(str(output_path))

    # Verify the quantized model
    model_size = sum(
        f.stat().st_size for f in output_path.rglob("*.safetensors")
    ) / 1e9
    print(f"\nGPTQ quantization complete!")
    print(f"  Model size: {model_size:.1f} GB")
    print(f"  Saved to: {output_path}")
    return True


def test_gptq(model_path: Path) -> bool:
    """Test GPTQ model with a quick inference."""
    try:
        from auto_gptq import AutoGPTQForCausalLM  # type: ignore[import]
        from transformers import AutoTokenizer
        import torch

        print(f"\nTesting GPTQ model: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        model = AutoGPTQForCausalLM.from_quantized(
            str(model_path),
            device="cuda:0",
            use_safetensors=True,
        )

        prompt = "<|user|>\nWhat is the recommended user count threshold for three-tier vs collapsed core?<|end|>\n<|assistant|>"
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False)

        text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        print(f"Test response: {text[:300]}...")
        print("GPTQ test: PASSED")
        return True
    except Exception as e:
        print(f"GPTQ test failed: {e}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GPTQ Quantization for Phi-4",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model",
        default=str(ROOT / "models" / "phi4-network-architect"),
        help="Path to fine-tuned model",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "models" / "phi4-gptq"),
        help="Output directory for GPTQ model",
    )
    parser.add_argument("--bits", type=int, default=4, choices=[2, 3, 4, 8])
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument(
        "--calibration-data",
        default=str(ROOT / "generated_data" / "training_data.jsonl"),
        help="Path to training data for calibration",
    )
    parser.add_argument("--calibration-samples", type=int, default=128)
    parser.add_argument("--test", action="store_true", help="Test after quantization")
    args = parser.parse_args()

    model_path = Path(args.model)
    output_path = Path(args.output)
    cal_path = Path(args.calibration_data) if args.calibration_data else None

    ok = quantize_gptq(
        model_path=model_path,
        output_path=output_path,
        bits=args.bits,
        group_size=args.group_size,
        calibration_data_path=cal_path if cal_path and cal_path.exists() else None,
        calibration_samples=args.calibration_samples,
    )

    if ok and args.test:
        test_gptq(output_path)

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
