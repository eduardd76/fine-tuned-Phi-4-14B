"""
GGUF Quantization Script
Converts fine-tuned Phi-4 model to GGUF format for CPU inference.

Usage:
    python deployment/quantize_gguf.py \
        --model models/phi4-network-architect \
        --output models/phi4-q4.gguf \
        --quant-type Q4_K_M

    # All quantization types
    python deployment/quantize_gguf.py --model models/phi4-network-architect --all

Quantization types by use case:
    Q4_K_M  - Recommended: best quality/size balance for CPU inference
    Q5_K_M  - Higher quality, larger file (~6.5 GB)
    Q8_0    - Near-lossless, largest (~11 GB)
    Q3_K_M  - Smallest usable (~4.5 GB), some quality loss
    F16     - Full precision float16 (~26 GB), max quality
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent.parent

QUANT_TYPES = {
    "Q4_K_M": {"size_gb": 5.5, "quality": "good", "description": "Recommended - best balance"},
    "Q5_K_M": {"size_gb": 6.5, "quality": "better", "description": "Higher quality, 20% larger"},
    "Q8_0":   {"size_gb": 11.0, "quality": "best", "description": "Near-lossless quantization"},
    "Q3_K_M": {"size_gb": 4.5, "quality": "ok", "description": "Smallest usable format"},
    "F16":    {"size_gb": 26.0, "quality": "lossless", "description": "Full precision"},
}


def check_dependencies() -> bool:
    """Check if llama.cpp is available."""
    try:
        import llama_cpp  # type: ignore[import]
        print(f"llama-cpp-python: OK ({llama_cpp.__version__})")
        return True
    except ImportError:
        print("llama-cpp-python not installed.")
        print("Install: pip install llama-cpp-python")
        print("With CUDA: CMAKE_ARGS='-DLLAMA_CUDA=on' pip install llama-cpp-python")
        return False


def merge_lora_if_needed(model_path: Path) -> Path:
    """
    If model_path contains LoRA adapters, merge them into the base model first.
    Returns path to merged model.
    """
    lora_adapter = model_path / "lora_adapters"
    if not lora_adapter.exists():
        return model_path

    merged_path = model_path.parent / (model_path.name + "_merged")
    if merged_path.exists():
        print(f"Using existing merged model: {merged_path}")
        return merged_path

    print("LoRA adapters detected - merging into base model...")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel  # type: ignore[import]
        import torch

        print("Loading base model...")
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        base_model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=torch.float16,
            device_map="cpu",
        )

        print("Loading and merging LoRA adapters...")
        model = PeftModel.from_pretrained(base_model, str(lora_adapter))
        model = model.merge_and_unload()

        print(f"Saving merged model to: {merged_path}")
        merged_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(merged_path))
        tokenizer.save_pretrained(str(merged_path))
        print("Merge complete!")
        return merged_path

    except Exception as e:
        print(f"LoRA merge failed: {e}")
        print("Proceeding with unmerged model (quality may be lower)")
        return model_path


def quantize_to_gguf(
    model_path: Path,
    output_path: Path,
    quant_type: str = "Q4_K_M",
) -> bool:
    """
    Convert model to GGUF format using llama.cpp convert scripts.

    Returns True on success.
    """
    print(f"\nQuantizing to GGUF ({quant_type})")
    print(f"  Input:  {model_path}")
    print(f"  Output: {output_path}")
    print(f"  Expected size: ~{QUANT_TYPES.get(quant_type, {}).get('size_gb', '?')} GB")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Try using Unsloth's built-in GGUF export (preferred for Phi-4)
    try:
        from unsloth import FastLanguageModel  # type: ignore[import]

        print("Using Unsloth GGUF export...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            str(model_path),
            max_seq_length=4096,
            load_in_4bit=False,  # Load in full precision for quantization
        )

        model.save_pretrained_gguf(
            str(output_path.with_suffix("")),  # Unsloth adds .gguf
            tokenizer,
            quantization_method=quant_type.lower(),
        )
        print(f"GGUF saved to: {output_path}")
        return True

    except ImportError:
        print("Unsloth not available, trying llama.cpp convert script...")

    # Fallback: use llama.cpp Python bindings
    try:
        import llama_cpp  # type: ignore[import]
        llama_cpp_dir = Path(llama_cpp.__file__).parent

        # Find convert script
        convert_script = llama_cpp_dir / "llama_cpp" / "server" / "convert.py"
        if not convert_script.exists():
            # Try common locations
            for candidate in [
                Path("llama.cpp/convert_hf_to_gguf.py"),
                Path("/usr/local/lib/python3.11/dist-packages/llama_cpp/convert.py"),
            ]:
                if candidate.exists():
                    convert_script = candidate
                    break

        if convert_script.exists():
            # Step 1: Convert to F16 GGUF
            f16_path = output_path.parent / f"{output_path.stem}_f16.gguf"
            cmd_convert = [
                sys.executable, str(convert_script),
                str(model_path),
                "--outfile", str(f16_path),
                "--outtype", "f16",
            ]
            print(f"Converting to F16: {' '.join(cmd_convert)}")
            result = subprocess.run(cmd_convert, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Conversion failed: {result.stderr}")
                return False

            # Step 2: Quantize F16 to target type
            cmd_quant = ["llama-quantize", str(f16_path), str(output_path), quant_type]
            print(f"Quantizing to {quant_type}: {' '.join(cmd_quant)}")
            result = subprocess.run(cmd_quant, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Quantization failed: {result.stderr}")
                print("Try installing llama.cpp: https://github.com/ggerganov/llama.cpp")
                return False

            # Cleanup F16 intermediate
            if f16_path.exists():
                f16_path.unlink()

            print(f"GGUF saved to: {output_path}")
            return True

    except Exception as e:
        print(f"llama.cpp quantization failed: {e}")

    print("\nManual quantization instructions:")
    print("1. Install llama.cpp: git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp && make")
    print(f"2. python convert_hf_to_gguf.py {model_path} --outtype f16 --outfile /tmp/phi4-f16.gguf")
    print(f"3. ./llama-quantize /tmp/phi4-f16.gguf {output_path} {quant_type}")
    return False


def test_gguf(gguf_path: Path, prompt: str = "What is spine-leaf architecture?") -> bool:
    """Quick test of the GGUF file."""
    try:
        from llama_cpp import Llama  # type: ignore[import]

        print(f"\nTesting GGUF: {gguf_path}")
        llm = Llama(model_path=str(gguf_path), n_ctx=512, verbose=False)
        output = llm(prompt, max_tokens=100, temperature=0.1)
        text = output["choices"][0]["text"]
        print(f"Test response (100 tokens): {text[:200]}...")
        print("GGUF test: PASSED")
        return True
    except Exception as e:
        print(f"GGUF test failed: {e}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quantize Phi-4 to GGUF",
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
        default=str(ROOT / "models" / "phi4-q4.gguf"),
        help="Output GGUF file path",
    )
    parser.add_argument(
        "--quant-type",
        default="Q4_K_M",
        choices=list(QUANT_TYPES.keys()),
        help="Quantization type",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate all quantization types",
    )
    parser.add_argument("--test", action="store_true", help="Test after quantization")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        print("Train first: python fine_tuning/train.py")
        sys.exit(1)

    print(f"Input model: {model_path}")
    print("\nQuantization options:")
    for qt, info in QUANT_TYPES.items():
        print(f"  {qt}: ~{info['size_gb']}GB - {info['description']}")

    # Merge LoRA if needed
    model_path = merge_lora_if_needed(model_path)

    if args.all:
        success_count = 0
        for qt in QUANT_TYPES:
            output = ROOT / "models" / f"phi4-{qt.lower().replace('_', '-')}.gguf"
            ok = quantize_to_gguf(model_path, output, qt)
            if ok:
                success_count += 1
                if args.test:
                    test_gguf(output)
        print(f"\nCompleted: {success_count}/{len(QUANT_TYPES)} quantization types")
    else:
        output_path = Path(args.output)
        ok = quantize_to_gguf(model_path, output_path, args.quant_type)
        if ok and args.test:
            test_gguf(output_path)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
