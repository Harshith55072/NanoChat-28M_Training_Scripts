"""
Quantizes the trained model to int8 (dynamic quantization) for fast CPU inference,
and also saves a clean fp32 safetensors copy (the Hugging-Face-standard weight format).

Requires: safetensors
    pip install safetensors

Usage:
    python inference/quantize.py --checkpoint checkpoints/final.pt
"""

import os
import sys
import argparse

import torch
import torch.nn as nn
from safetensors.torch import save_model

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.model import ChatGPTMini, ModelConfig


def load_model_from_checkpoint(ckpt_path, device="cpu"):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["config"]
    model = ChatGPTMini(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/final.pt")
    ap.add_argument("--out_dir", default="checkpoints")
    args = ap.parse_args()

    print(f"Loading checkpoint from {args.checkpoint} ...")
    model, cfg = load_model_from_checkpoint(args.checkpoint, device="cpu")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  loaded model: {n_params:,} params ({n_params/1e6:.2f}M)")

    # ---- 1. save a clean fp32 safetensors copy (standard HF weight format) ----
    fp32_path = os.path.join(args.out_dir, "model_fp32.safetensors")
    save_model(model, fp32_path)
    fp32_size_mb = os.path.getsize(fp32_path) / (1024 * 1024)
    print(f"Saved fp32 safetensors -> {fp32_path} ({fp32_size_mb:.1f} MB)")

    # ---- 2. dynamic int8 quantization (quantizes nn.Linear layers -- where ----
    # ----    almost all the params and compute live in a transformer) ----
    print("Applying dynamic int8 quantization to Linear layers...")
    quantized_model = torch.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )

    # dynamic-quantized modules don't serialize cleanly to safetensors (they're not
    # plain tensors), so we save the whole quantized module via torch.save instead.
    # This is standard practice for PyTorch dynamic quantization.
    quant_path = os.path.join(args.out_dir, "model_int8_dynamic.pt")
    torch.save({"model": quantized_model, "config": cfg}, quant_path)
    quant_size_mb = os.path.getsize(quant_path) / (1024 * 1024)
    print(f"Saved int8 quantized model -> {quant_path} ({quant_size_mb:.1f} MB)")

    print(f"\nSize comparison: fp32 {fp32_size_mb:.1f} MB -> int8 {quant_size_mb:.1f} MB "
          f"({fp32_size_mb/quant_size_mb:.2f}x smaller)")
    print("\nNote: model_int8_dynamic.pt contains the full quantized module (not just a "
          "state_dict) -- load it with torch.load(path)['model'] directly, no need to "
          "rebuild the architecture first.")
