"""
Production Inference Script
Phi-4 Network Architect

Supports: fine-tuned LoRA model, GPTQ quantized, GGUF (llama-cpp)

Usage:
    # Interactive mode
    python deployment/inference.py --model models/phi4-network-architect

    # Single query
    python deployment/inference.py --model models/phi4-network-architect \
        --prompt "Design network for 5000 users, PCI-DSS, 99.99% uptime"

    # GGUF model (CPU)
    python deployment/inference.py --model models/phi4-q4.gguf --backend gguf

    # GPTQ model (GPU, faster)
    python deployment/inference.py --model models/phi4-gptq --backend gptq

    # FastAPI server mode
    python deployment/inference.py --model models/phi4-network-architect --serve --port 8080
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

# Add project root
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

SYSTEM_PROMPT = """You are a Virtual Network Architect with CCDE-level expertise.
You design enterprise networks, data centers, and WAN architectures.
You ground every recommendation in industry standards and vendor documentation.
Always use <think> tags to show your structured reasoning before answering."""


# ─────────────────────────────────────────────────────────────
# Backend loaders
# ─────────────────────────────────────────────────────────────

class HFInferenceBackend:
    """HuggingFace transformers inference (LoRA or full model)."""

    def __init__(self, model_path: str, load_in_4bit: bool = True) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading model from {model_path} on {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        kwargs: dict[str, Any] = {
            "device_map": "auto" if self.device == "cuda" else "cpu",
        }
        if load_in_4bit and self.device == "cuda":
            from transformers import BitsAndBytesConfig
            import torch
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        self.model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        self.model.eval()
        print("Model loaded.")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        import torch

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        try:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            text = f"<|system|>\n{SYSTEM_PROMPT}<|end|>\n<|user|>\n{prompt}<|end|>\n<|assistant|>"

        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)


class GGUFInferenceBackend:
    """llama-cpp-python inference from GGUF file (CPU/GPU)."""

    def __init__(self, model_path: str, n_gpu_layers: int = -1, n_ctx: int = 4096) -> None:
        from llama_cpp import Llama

        print(f"Loading GGUF model: {model_path}")
        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            verbose=False,
        )
        print("GGUF model loaded.")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        full_prompt = (
            f"<|system|>\n{SYSTEM_PROMPT}<|end|>\n"
            f"<|user|>\n{prompt}<|end|>\n"
            f"<|assistant|>\n"
        )
        output = self.llm(
            full_prompt,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=["<|end|>", "<|user|>"],
        )
        return output["choices"][0]["text"]


class GPTQInferenceBackend:
    """AutoGPTQ inference (4-bit GPU)."""

    def __init__(self, model_path: str) -> None:
        from auto_gptq import AutoGPTQForCausalLM
        from transformers import AutoTokenizer
        import torch

        print(f"Loading GPTQ model: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoGPTQForCausalLM.from_quantized(
            model_path,
            device="cuda:0",
            use_safetensors=True,
        )
        self.model.eval()
        self.device = "cuda"
        print("GPTQ model loaded.")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        import torch

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        try:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            text = f"<|system|>\n{SYSTEM_PROMPT}<|end|>\n<|user|>\n{prompt}<|end|>\n<|assistant|>"

        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
            )

        generated = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)


# ─────────────────────────────────────────────────────────────
# Inference engine
# ─────────────────────────────────────────────────────────────

def parse_output(raw: str) -> dict[str, Any]:
    """
    Parse model output into structured components.

    Returns:
        {
            "reasoning": "<think>...</think>",
            "answer": "Final design...",
            "sources": [...],
            "has_think_block": bool,
            "reasoning_steps": int,
        }
    """
    # Extract <think> block
    think_match = re.search(r"<think>(.*?)</think>", raw, re.DOTALL)
    reasoning = think_match.group(0) if think_match else ""
    think_content = think_match.group(1) if think_match else ""

    # Extract answer (everything after </think>)
    answer = raw
    if think_match:
        answer = raw[think_match.end():].strip()

    # Extract source citations (pattern: [Source: ...] or "Source:" lines)
    sources = re.findall(
        r"\[Source:\s*([^\]]+)\]|Source:\s*([^\n]+)",
        raw,
        re.IGNORECASE,
    )
    unique_sources = list({(s[0] or s[1]).strip() for s in sources if s[0] or s[1]})

    # Count reasoning steps
    step_count = len(re.findall(r"^Step\s+\d+:", think_content, re.MULTILINE))

    return {
        "reasoning": reasoning,
        "answer": answer,
        "sources": unique_sources,
        "has_think_block": bool(think_match),
        "reasoning_steps": step_count,
    }


def infer(
    prompt: str,
    backend: Any,
    max_new_tokens: int = 2048,
    temperature: float = 0.7,
) -> dict[str, Any]:
    """
    Run inference and return structured output.

    Args:
        prompt: User query
        backend: Inference backend instance
        max_new_tokens: Max tokens to generate
        temperature: Sampling temperature

    Returns:
        {
            "reasoning": "<think>...</think>",
            "answer": "Final design...",
            "sources": ["Book Ch3", "PCI-DSS 4.0.1 §1.2"],
            "latency_ms": 2847,
            "has_think_block": bool,
            "reasoning_steps": int,
        }
    """
    t0 = time.perf_counter()
    raw = backend.generate(prompt, max_new_tokens=max_new_tokens, temperature=temperature)
    latency_ms = int((time.perf_counter() - t0) * 1000)

    result = parse_output(raw)
    result["latency_ms"] = latency_ms
    result["raw"] = raw
    return result


# ─────────────────────────────────────────────────────────────
# FastAPI server
# ─────────────────────────────────────────────────────────────

def serve(backend: Any, port: int = 8080) -> None:
    """Start FastAPI inference server."""
    try:
        from fastapi import FastAPI
        from pydantic import BaseModel
        import uvicorn
    except ImportError:
        print("Install server deps: pip install fastapi uvicorn")
        sys.exit(1)

    app = FastAPI(
        title="Phi-4 Network Architect API",
        description="CCDE-level network design inference",
    )

    class InferenceRequest(BaseModel):
        prompt: str
        max_new_tokens: int = 2048
        temperature: float = 0.7

    class InferenceResponse(BaseModel):
        reasoning: str
        answer: str
        sources: list[str]
        latency_ms: int
        has_think_block: bool
        reasoning_steps: int

    @app.get("/health")
    async def health():
        return {"status": "ok", "model": "phi4-network-architect"}

    @app.post("/infer", response_model=InferenceResponse)
    async def run_inference(req: InferenceRequest):
        result = infer(req.prompt, backend, req.max_new_tokens, req.temperature)
        return InferenceResponse(**{k: v for k, v in result.items() if k != "raw"})

    print(f"Starting server on http://0.0.0.0:{port}")
    print(f"API docs: http://localhost:{port}/docs")
    uvicorn.run(app, host="0.0.0.0", port=port)


# ─────────────────────────────────────────────────────────────
# Interactive mode
# ─────────────────────────────────────────────────────────────

def interactive(backend: Any) -> None:
    """Run interactive REPL."""
    print("\n" + "=" * 60)
    print("Phi-4 Network Architect - Interactive Mode")
    print("Type 'quit' or Ctrl+C to exit")
    print("=" * 60 + "\n")

    while True:
        try:
            prompt = input("Query> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not prompt:
            continue
        if prompt.lower() in {"quit", "exit", "q"}:
            break

        print("\nGenerating...", flush=True)
        result = infer(prompt, backend)

        print(f"\n{'─' * 60}")
        if result["has_think_block"]:
            print(f"[Reasoning: {result['reasoning_steps']} steps]")
            print(result["reasoning"])
            print()
        print("[Answer]")
        print(result["answer"])
        if result["sources"]:
            print(f"\n[Sources: {', '.join(result['sources'][:5])}]")
        print(f"\n[Latency: {result['latency_ms']}ms]")
        print(f"{'─' * 60}\n")


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phi-4 Network Architect Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model", required=True, help="Path to model (dir or .gguf file)")
    parser.add_argument(
        "--backend",
        choices=["hf", "gguf", "gptq"],
        default=None,
        help="Inference backend (auto-detected from model path if not set)",
    )
    parser.add_argument("--prompt", default=None, help="Single query (non-interactive)")
    parser.add_argument("--serve", action="store_true", help="Start FastAPI server")
    parser.add_argument("--port", type=int, default=8080, help="Server port")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization (HF backend)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference (GGUF only)")
    args = parser.parse_args()

    # Auto-detect backend
    backend_type = args.backend
    if backend_type is None:
        if args.model.endswith(".gguf"):
            backend_type = "gguf"
        elif Path(args.model).is_dir() and (Path(args.model) / "quantize_config.json").exists():
            backend_type = "gptq"
        else:
            backend_type = "hf"

    # Load backend
    if backend_type == "gguf":
        n_gpu_layers = 0 if args.cpu else -1
        backend = GGUFInferenceBackend(args.model, n_gpu_layers=n_gpu_layers)
    elif backend_type == "gptq":
        backend = GPTQInferenceBackend(args.model)
    else:
        backend = HFInferenceBackend(args.model, load_in_4bit=not args.no_4bit)

    # Run in requested mode
    if args.serve:
        serve(backend, port=args.port)
    elif args.prompt:
        result = infer(args.prompt, backend, args.max_tokens, args.temperature)
        print(json.dumps({k: v for k, v in result.items() if k != "raw"}, indent=2))
    else:
        interactive(backend)


if __name__ == "__main__":
    main()
