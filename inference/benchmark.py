"""
Benchmarks fp32 vs int8-quantized model speed on CPU -- this is the real test of
whether the model is fast enough for real-time chat generation.

Usage:
    python inference/benchmark.py
"""

import os
import sys
import time
import argparse

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.model import ChatGPTMini, ModelConfig
from tokenizers import Tokenizer

DEFAULT_TOKENIZER_PATH = "../data/NanoChat-28M_model_data/tokenizer/tokenizer.json"


def load_fp32_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["config"]
    model = ChatGPTMini(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


def load_quantized_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    return ckpt["model"], ckpt["config"]


@torch.no_grad()
def benchmark_model(model, tokenizer, bos_id, chat_id, eos_id, n_generations=30, max_new_tokens=30, batch_size=1):
    n_batches = max(1, n_generations // batch_size)
    prompt = torch.tensor([[bos_id, chat_id]] * batch_size, dtype=torch.long)

    # warmup (first call is often slower -- don't count it)
    _ = model.generate(prompt, max_new_tokens=10, eos_token_id=eos_id)

    t0 = time.time()
    total_sequences = 0
    total_tokens = 0
    for _ in range(n_batches):
        out = model.generate(prompt, max_new_tokens=max_new_tokens, eos_token_id=eos_id)
        total_sequences += out.shape[0]
        total_tokens += (out.shape[1] - prompt.shape[1]) * out.shape[0]
    elapsed = time.time() - t0

    gens_per_sec = total_sequences / elapsed
    tokens_per_sec = total_tokens / elapsed
    return gens_per_sec, tokens_per_sec, elapsed


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer_path", default=DEFAULT_TOKENIZER_PATH)
    ap.add_argument("--fp32_checkpoint", default="checkpoints/final.pt")
    ap.add_argument("--quant_checkpoint", default="checkpoints/model_int8_dynamic.pt")
    ap.add_argument("--n_generations", type=int, default=64)
    ap.add_argument("--max_new_tokens", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    torch.set_num_threads(os.cpu_count())
    print(f"CPU threads available to PyTorch: {torch.get_num_threads()}")

    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    bos_id = tokenizer.token_to_id("<bos>")
    chat_id = tokenizer.token_to_id("<chat>")
    eos_id = tokenizer.token_to_id("<eos>")

    print(f"\nBenchmarking: {args.n_generations} generations, batch_size={args.batch_size}, "
          f"{args.max_new_tokens} tokens each\n")

    print("== fp32 model (batched) ==")
    fp32_model, _ = load_fp32_model(args.fp32_checkpoint)
    gps, tps, elapsed = benchmark_model(fp32_model, tokenizer, bos_id, chat_id, eos_id,
                                         args.n_generations, args.max_new_tokens, args.batch_size)
    print(f"  {gps:.2f} generations/sec | {tps:.1f} tokens/sec | {elapsed:.2f}s total\n")

    print("== int8 quantized model (batched) ==")
    quant_model, _ = load_quantized_model(args.quant_checkpoint)
    gps_q, tps_q, elapsed_q = benchmark_model(quant_model, tokenizer, bos_id, chat_id, eos_id,
                                                args.n_generations, args.max_new_tokens, args.batch_size)
    print(f"  {gps_q:.2f} generations/sec | {tps_q:.1f} tokens/sec | {elapsed_q:.2f}s total\n")

    speedup = tps_q / tps if tps > 0 else float("nan")
    print(f"Speedup from quantization: {speedup:.2f}x")
    print(f"\nFor context: your use case needs 'many short comments/sec' -- at "
          f"{gps_q:.1f} generations/sec on int8, that's roughly what you'd have available "
          f"to work with for live simulated chat volume.")
