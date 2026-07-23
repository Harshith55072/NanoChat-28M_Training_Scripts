"""
Interactive terminal demo -- simulates the live chat / superchat wall reacting to
whatever you type as "what the streamer just said."

Run from the NanoChat-28M_Training_Scripts project root:
    python inference/interface.py

At each prompt, type a line as if you were the streamer (e.g. "let's try the new boss
fight tonight") and press Enter. The script will print a burst of simulated chat
comments, occasionally topically nudged by a keyword pulled from what you typed, plus
an occasional simulated superchat. Type 'quit' or 'exit' to stop.

This demo runs the fp32 model directly in Python for simplicity/portability (see
project.md -- fp32 was the final decision over int8, KV-cache is what actually matters
for speed). No GPU required; this is deliberately testing the real CPU deployment path.
"""

import os
import sys
import time
import random
import argparse

import torch
from tokenizers import Tokenizer
from safetensors.torch import load_model

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.model import ChatGPTMini, ModelConfig
from keyword_utils import extract_keywords, build_seed_prompt, pick_keyword_or_none

DEFAULT_TOKENIZER_PATH = "../data/NanoChat-28M_model_data/tokenizer/tokenizer.json"
DEFAULT_CHECKPOINT = "checkpoints/model_fp32.safetensors"

# sampling "personality" presets -- calm/lurker-ish through chaotic/hype-spammer, cheap
# way to fake per-comment personality variety without the model modeling personas itself
PERSONA_PRESETS = [
    {"temperature": 0.7, "top_k": 20, "top_p": 0.85},   # calm / low-key
    {"temperature": 0.9, "top_k": 40, "top_p": 0.90},   # typical chatter
    {"temperature": 1.1, "top_k": 60, "top_p": 0.95},   # excitable
    {"temperature": 1.3, "top_k": 80, "top_p": 0.97},   # unhinged hype spammer
]

FAKE_USERNAME_PARTS = (
    ["Shadow", "Pixel", "Turbo", "Salty", "Cosmic", "Lazy", "Feral", "Cursed", "Golden",
     "Silent", "Neon", "Rogue", "Sleepy", "Toxic", "Blessed"],
    ["Wolf", "Noodle", "Gremlin", "Potato", "Ghost", "Bandit", "Muffin", "Viper",
     "Goblin", "Otter", "Yeti", "Panda", "Raccoon", "Falcon", "Newt"],
)


def fake_username():
    a = random.choice(FAKE_USERNAME_PARTS[0])
    b = random.choice(FAKE_USERNAME_PARTS[1])
    n = random.randint(1, 9999)
    return f"{a}{b}{n}"


def load_tokenizer_and_model(tokenizer_path, checkpoint_path, device="cpu"):
    tokenizer = Tokenizer.from_file(tokenizer_path)
    pad_id = tokenizer.token_to_id("<pad>")
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")
    chat_id = tokenizer.token_to_id("<chat>")
    superchat_id = tokenizer.token_to_id("<superchat>")

    cfg = ModelConfig(vocab_size=tokenizer.get_vocab_size(), pad_token_id=pad_id)
    model = ChatGPTMini(cfg)
    load_model(model, checkpoint_path)  # safetensors loader, handles tied weights
    model.to(device).eval()

    ids = {"pad": pad_id, "bos": bos_id, "eos": eos_id, "chat": chat_id, "superchat": superchat_id}
    return tokenizer, model, ids


def generate_batch(model, prompt, persona, eos_id, max_new_tokens):
    return model.generate(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=persona["temperature"],
        top_k=persona["top_k"],
        top_p=persona["top_p"],
        eos_token_id=eos_id,
    )


def run_turn(tokenizer, model, ids, context_window, args):
    keywords = extract_keywords(context_window, top_k=3)

    all_lines = []  # (label, text) pairs to print, in generation order

    # ---- regular chat, split across persona presets so we get variety ----
    remaining = args.messages_per_turn
    n_personas = len(PERSONA_PRESETS)
    for i, persona in enumerate(PERSONA_PRESETS):
        group_size = remaining // (n_personas - i)
        remaining -= group_size
        if group_size <= 0:
            continue

        keyword = pick_keyword_or_none(keywords, seed_probability=0.55)
        prompt = build_seed_prompt(tokenizer, ids["bos"], ids["chat"], keyword, batch_size=group_size)
        out = generate_batch(model, prompt, persona, ids["eos"], args.max_new_tokens)

        for row in out:
            text = tokenizer.decode(row.tolist(), skip_special_tokens=True).strip()
            if text:
                all_lines.append(("chat", text))

    # ---- occasional superchat ----
    if random.random() < args.superchat_chance:
        keyword = pick_keyword_or_none(keywords, seed_probability=0.7)  # superchats lean more on-topic
        prompt = build_seed_prompt(tokenizer, ids["bos"], ids["superchat"], keyword, batch_size=1)
        persona = random.choice(PERSONA_PRESETS[:2])  # superchats stay more coherent, less chaotic
        out = generate_batch(model, prompt, persona, ids["eos"], args.max_new_tokens + 10)
        text = tokenizer.decode(out[0].tolist(), skip_special_tokens=True).strip()
        if text:
            all_lines.append(("superchat", text))

    random.shuffle(all_lines)
    return all_lines


def print_lines(lines, delay):
    for label, text in lines:
        if label == "superchat":
            tier = random.choice(["$2", "$5", "$10", "$20", "$50"])
            print(f"  \033[93m[SUPERCHAT {tier}] {fake_username()}:\033[0m {text}")
        else:
            print(f"  {fake_username()}: {text}")
        if delay > 0:
            time.sleep(delay)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer_path", default=DEFAULT_TOKENIZER_PATH)
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--messages_per_turn", type=int, default=12)
    ap.add_argument("--max_new_tokens", type=int, default=20)
    ap.add_argument("--superchat_chance", type=float, default=0.20)
    ap.add_argument("--delay", type=float, default=0.08, help="seconds between printed lines, for a 'live' feel")
    ap.add_argument("--context_lines", type=int, default=2, help="how many recent transcript lines to keep as context")
    args = ap.parse_args()

    print("Loading model...")
    tokenizer, model, ids = load_tokenizer_and_model(args.tokenizer_path, args.checkpoint, args.device)
    print("Ready.\n")
    print("Type a line as if you're the streamer, and watch the chat react.")
    print("Type 'quit' or 'exit' to stop.\n")

    context_window = []

    while True:
        try:
            line = input("streamer> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if line.lower() in ("quit", "exit"):
            print("Exiting.")
            break
        if not line:
            continue

        context_window.append(line)
        context_window = context_window[-args.context_lines:]

        t0 = time.time()
        lines = run_turn(tokenizer, model, ids, context_window, args)
        elapsed = time.time() - t0

        print()
        print_lines(lines, args.delay)
        print(f"\n  ({len(lines)} messages generated in {elapsed:.2f}s)\n")


if __name__ == "__main__":
    main()
