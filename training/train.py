"""
Training script for the chat/superchat generator.

Run from the NanoChat-28M_Training_Scripts project root:
    python training/train.py

Key design decisions (see project.md for full context):
- Superchat class imbalance (634 rows vs 3.1M chat rows) is fixed via a *sampling ratio*,
  not by duplicating files. Each training example has a `superchat_ratio` chance of being
  drawn from the superchat pool regardless of its tiny size -- see MixedChatDataset.
- Each sequence is: <bos> <chat|superchat> [text tokens...] <eos>, so the model always
  knows which mode it's generating in (v1 has no other context conditioning -- see
  project.md "Context/relevance problem" section for why).
- Right-padding + causal attention means padded positions are automatically never
  attended to by real tokens, so no extra attention mask is needed -- only the loss
  needs to ignore pad positions (handled via ignore_index in the model's forward()).
"""

import os
import sys
import json
import time
import random
import argparse

import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer

# allow "from model.model import ..." when running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.model import ChatGPTMini, ModelConfig


# ---------------- paths (relative to this project's root, pointing at the data project) ----------------
DEFAULT_TOKENIZER_PATH = "../data/NanoChat-28M_model_data/tokenizer/tokenizer.json"
DEFAULT_CHAT_PATH = "../data/NanoChat-28M_model_data/processed/train_chat.jsonl"
DEFAULT_SUPERCHAT_PATH = "../data/NanoChat-28M_model_data/processed/train_superchat.jsonl"


# ---------------- data loading ----------------

def load_texts(path):
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            t = row.get("text", "").strip()
            if t:
                texts.append(t)
    return texts


def tokenize_all(tokenizer, texts, mode_token_id, bos_id, eos_id, max_seq_len):
    """Tokenizes a list of raw strings into [bos, mode, ...text..., eos] id lists,
    truncated to max_seq_len. Uses batch encoding for speed."""
    encodings = tokenizer.encode_batch(texts)
    sequences = []
    budget = max_seq_len - 3  # room for bos + mode token + eos
    for enc in encodings:
        ids = enc.ids[:budget]
        seq = [bos_id, mode_token_id] + ids + [eos_id]
        sequences.append(seq)
    return sequences


class MixedChatDataset(Dataset):
    """Randomly draws from the chat pool or the (much smaller) superchat pool according
    to a fixed ratio, so superchat gets meaningfully represented in training regardless
    of its tiny raw file size."""

    def __init__(self, chat_seqs, superchat_seqs, superchat_ratio=0.12, epoch_len=None):
        self.chat_seqs = chat_seqs
        self.superchat_seqs = superchat_seqs
        self.superchat_ratio = superchat_ratio
        self.epoch_len = epoch_len or len(chat_seqs)

    def __len__(self):
        return self.epoch_len

    def __getitem__(self, idx):
        if random.random() < self.superchat_ratio:
            seq = random.choice(self.superchat_seqs)
        else:
            seq = random.choice(self.chat_seqs)
        return torch.tensor(seq, dtype=torch.long)


def collate_fn(batch, pad_id):
    max_len = max(len(seq) for seq in batch)
    padded = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    for i, seq in enumerate(batch):
        padded[i, :len(seq)] = seq
    return padded


# ---------------- lr schedule ----------------

def lr_lambda(step, warmup_steps, total_steps):
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(progress, 1.0)
    return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159265)).item())


# ---------------- sample generation (for watching progress live) ----------------

@torch.no_grad()
def print_samples(model, tokenizer, device, bos_id, chat_id, superchat_id, eos_id):
    model.eval()
    for label, mode_id in [("chat", chat_id), ("superchat", superchat_id)]:
        prompt = torch.tensor([[bos_id, mode_id]], dtype=torch.long, device=device)
        out = model.generate(prompt, max_new_tokens=30, temperature=0.9, top_k=40, top_p=0.9, eos_token_id=eos_id)
        text = tokenizer.decode(out[0].tolist(), skip_special_tokens=True)
        print(f"    [{label}] {text!r}")
    model.train()


# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer_path", default=DEFAULT_TOKENIZER_PATH)
    ap.add_argument("--chat_path", default=DEFAULT_CHAT_PATH)
    ap.add_argument("--superchat_path", default=DEFAULT_SUPERCHAT_PATH)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--total_steps", type=int, default=30000)
    ap.add_argument("--warmup_steps", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--superchat_ratio", type=float, default=0.12)
    ap.add_argument("--log_interval", type=int, default=50)
    ap.add_argument("--sample_interval", type=int, default=500)
    ap.add_argument("--ckpt_interval", type=int, default=2000)
    ap.add_argument("--ckpt_dir", default="checkpoints")
    ap.add_argument("--resume", default=None, help="path to a checkpoint .pt file to resume from")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cpu":
        print("  [warn] no GPU detected -- training will be much slower than on your 4050.")

    print(f"Loading tokenizer from {args.tokenizer_path} ...")
    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    pad_id = tokenizer.token_to_id("<pad>")
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")
    chat_id = tokenizer.token_to_id("<chat>")
    superchat_id = tokenizer.token_to_id("<superchat>")
    vocab_size = tokenizer.get_vocab_size()
    print(f"  vocab_size={vocab_size}, pad={pad_id}, bos={bos_id}, eos={eos_id}, "
          f"<chat>={chat_id}, <superchat>={superchat_id}")

    cfg = ModelConfig(vocab_size=vocab_size, pad_token_id=pad_id)

    print(f"Loading chat data from {args.chat_path} ...")
    chat_texts = load_texts(args.chat_path)
    print(f"  {len(chat_texts):,} chat rows")

    print(f"Loading superchat data from {args.superchat_path} ...")
    superchat_texts = load_texts(args.superchat_path)
    print(f"  {len(superchat_texts):,} superchat rows")

    print("Tokenizing (this can take a minute for the chat set)...")
    chat_seqs = tokenize_all(tokenizer, chat_texts, chat_id, bos_id, eos_id, cfg.max_seq_len)
    superchat_seqs = tokenize_all(tokenizer, superchat_texts, superchat_id, bos_id, eos_id, cfg.max_seq_len)
    print(f"  done. {len(chat_seqs):,} chat sequences, {len(superchat_seqs):,} superchat sequences")

    dataset = MixedChatDataset(chat_seqs, superchat_seqs, superchat_ratio=args.superchat_ratio)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_id),
        num_workers=0,  # 0 avoids Windows multiprocessing pickling headaches
    )

    model = ChatGPTMini(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params ({n_params/1e6:.2f}M)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    start_step = 0
    if args.resume:
        print(f"Resuming from {args.resume} ...")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"]
        print(f"  resumed at step {start_step}")

    os.makedirs(args.ckpt_dir, exist_ok=True)

    model.train()
    data_iter = iter(loader)
    t0 = time.time()
    running_loss = 0.0

    for step in range(start_step, args.total_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        batch = batch.to(device)
        input_ids = batch[:, :-1]
        targets = batch[:, 1:]

        lr_scale = lr_lambda(step, args.warmup_steps, args.total_steps)
        for g in optimizer.param_groups:
            g["lr"] = args.lr * lr_scale

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            logits, loss = model(input_ids, targets=targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

        if (step + 1) % args.log_interval == 0:
            avg_loss = running_loss / args.log_interval
            elapsed = time.time() - t0
            steps_per_sec = args.log_interval / elapsed
            print(f"step {step+1}/{args.total_steps} | loss {avg_loss:.4f} | "
                  f"lr {optimizer.param_groups[0]['lr']:.2e} | {steps_per_sec:.2f} steps/s")
            running_loss = 0.0
            t0 = time.time()

        if (step + 1) % args.sample_interval == 0:
            print("  -- sample generations --")
            print_samples(model, tokenizer, device, bos_id, chat_id, superchat_id, eos_id)

        if (step + 1) % args.ckpt_interval == 0:
            ckpt_path = os.path.join(args.ckpt_dir, f"step_{step+1}.pt")
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step + 1,
                "config": cfg,
            }, ckpt_path)
            print(f"  saved checkpoint -> {ckpt_path}")

    # final save
    final_path = os.path.join(args.ckpt_dir, "final.pt")
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": args.total_steps,
        "config": cfg,
    }, final_path)
    print(f"Training complete. Final checkpoint -> {final_path}")


if __name__ == "__main__":
    main()
