# AI Chat Simulator — Project Overview

## What this is
A very small language model that generates fake live-stream chat comments in real time,
to feed as background "audience" input to a separate, bigger AI VTuber model that is
actually streaming. Two output modes:

- **`chat`** — short, bursty, low-effort hype comments ("LMAOOO", "W stream", "no way 💀").
  High volume, low uniqueness needed, just needs to vaguely relate to what the big model
  is currently saying.
- **`superchat`** — rare, longer, more coherent paid-message-style comments/questions that
  the big model should acknowledge directly (e.g. "what's your favorite game to speedrun?").

Vague "different personalities" per commenter are faked cheaply via sampling param variation
(temperature/top-k/top-p) rather than the model explicitly modeling personas — see
"Decisions already made" below for why the persona field was dropped from the data.

## Hardware constraints
- Local laptop, RTX 4050, **6GB VRAM**. Training must fit in this.
- Final model must run **fast on CPU** at inference time (real-time chat generation),
  not just train-able on GPU. This drove the small param count decision.

## Repo split (two separate project folders)
1. **`NanoChat-28M_model_data`** (sibling folder — data collection & prep, DONE)
   `C:\Users\Lenovo\Documents\programing\data\NanoChat-28M_model_data`
   (renamed from `chat_ai_model_data` — official project name is now "NanoChat-28M")
2. **`NanoChat-28M_Training_Scripts`** (this folder — model build & training, IN PROGRESS)
   `C:\Users\Lenovo\Documents\programing\NanoChat-28M_Training_Scripts`
   (renamed from `chat_ai_model`)

Data is NOT copied into this folder — training scripts here reference the data
project via relative/absolute path:
```
../data/NanoChat-28M_model_data/processed/train_chat.jsonl
../data/NanoChat-28M_model_data/processed/train_superchat.jsonl
../data/NanoChat-28M_model_data/tokenizer/tokenizer.json
```
(adjust relative path depth depending on where a script actually lives)

## Data status (from chat_ai_model_data project)
- **Chat data**: `processed/train_chat.jsonl` — ~3.1M rows. Mostly real Twitch chat
  (Hugging Face dataset `lparkourer10/twitch_chat`, CC-BY-SA-4.0, usernames stripped),
  plus ~5k synthetic supplement. Cleaned (deduped, links/bot-commands stripped).
- **Superchat data**: `processed/train_superchat.jsonl` — 634 rows. Hand-written +
  template-generated (real superchat datasets don't publicly exist, had to bootstrap).
  Known limitation: small and somewhat template-flavored, will need oversampling
  during training so the model doesn't ignore this class given the ~5000:1 imbalance
  vs chat data.
- **Combined**: `processed/train_all.jsonl` (both modes, shuffled together).
- Schema: `{"mode": "chat"|"superchat", "text": "..."}` — **no persona field**
  (was random 1-20 with no real meaning, decided to drop it rather than keep noise
  or invent fake semantics).

## Tokenizer status: DONE
- Location: `NanoChat-28M_model_data/tokenizer/` (`tokenizer.json`, `vocab.json`, `merges.txt`)
- Type: byte-level BPE (GPT-2 style) — chosen so emojis/unicode never hit "unknown token"
- Vocab size: 8,000
- Special tokens: `<pad>`, `<bos>`, `<eos>`, `<unk>`, `<chat>`, `<superchat>`
  - `<chat>` / `<superchat>` are prepended to training sequences as mode markers so the
    model learns to condition generation on which mode it's in.
- Sanity check passed: mode tokens stay unsplit, common chat phrases tokenize compactly.
  Minor known inefficiency: some emojis split into 2 tokens instead of 1 — not fixed,
  not worth blocking on.

## Context/relevance problem (identified after tokenizer step)
Dataset is unpaired chat text only — no (streamer-context → reaction) pairs, since no
public dataset bundles timestamp-aligned transcript + chat. Real conditioning would need
either scraped/ASR-aligned VOD data (too heavy for this project) or a self-supervised
keyword-conditioning trick (extract fake "topic" from each comment, train on that, feed
real transcript keywords at inference).

**Decision: v1 uses runtime keyword injection instead of training-time conditioning.**
Model stays a pure style/vibe generator (no context input, no architecture change).
"Relevance" to the streamer is handled outside the model at inference time — cheap
keyword extraction from the big model's recent transcript, spliced/biased into which
generated lines get shown. Revisit self-supervised topic conditioning as a v2 upgrade
once the base model + pipeline is working end-to-end; existing data doesn't need to be
redone for that, just an added keyword-extraction pass over the same `text` field.

## Model plan: BUILT — `model/model.py`
- `ChatGPTMini`, a small pre-norm, decoder-only GPT-style transformer with causal
  self-attention, weight-tied input/output embeddings, and learned positional embeddings
  (max_seq_len=128; rotary/ALiBi deemed unnecessary since chat comments are short).
- **Actual chosen config (`ModelConfig` defaults)**: vocab_size=8000, d_model=448,
  n_layer=10, n_head=8 (head_dim=56), d_ff=1792 (4x d_model), dropout=0.1.
  This lands at **~27.7M params** per the file's own docstring/`count_params()` — the
  final dim (448) ended up higher than the "320-384" range floated earlier in planning.
- Includes a working `generate()` method (top-k + top-p/nucleus sampling, temperature,
  optional EOS stopping) and a `__main__` sanity check (param count + dummy forward pass).
- Not yet run/verified in this session — worth doing `python model/model.py` to confirm
  it still executes cleanly and reports ~27.7M params.

## Training status: DONE — `training/train.py`
- Loads tokenizer + `train_chat.jsonl`/`train_superchat.jsonl` from the sibling data
  project via relative path, tokenizes with `<bos> <chat|superchat> ... <eos>` framing.
- Class imbalance fix: `MixedChatDataset` draws from the superchat pool with fixed
  probability `superchat_ratio` (default 0.12) each `__getitem__` call, regardless of
  the pool's tiny size — not file duplication.
- AdamW, cosine LR schedule w/ warmup, grad clipping, AMP on CUDA, periodic sample
  generation + checkpointing, `--resume` support.
- **Actually run: 30,000 steps, ~65-70 min on the RTX 4050** (mixed precision,
  batch_size=128). Loss: 8.82 (random baseline) → 3.37 final, decreased steadily, no
  plateau issues. Sample generations at step 30000 showed real learned patterns: chat
  picked up @-mention reply style and Twitch-specific phrasing (emotes, bot message
  formats); superchat stayed close to training phrasing (634 examples is small, some
  memorization rather than full generalization is expected/acceptable for v1).
- Checkpoints saved every 2000 steps + `final.pt` in `checkpoints/` — **NOTE: gitignored,
  not backed up to GitHub, see Repo/backup section below.**

## Config files: NOT USED (by design)
- `configs/` only has `.gitkeep`. `train.py` takes all hyperparameters as CLI args with
  defaults baked in, so there's no separate config file being read. Fine as-is; could
  add a YAML/JSON config later if sweeping hyperparams becomes worthwhile.

## Inference plan: DONE (decision finalized)
- `inference/quantize.py`: produces `checkpoints/model_fp32.safetensors` (HF-standard
  weight format, correctly handles tied embedding/lm_head weights via
  `safetensors.torch.save_model`) and `checkpoints/model_int8_dynamic.pt` (int8 dynamic
  quantization) -- both kept in the codebase, but see decision below.
- **KV-caching implemented in `model/model.py`** (`CausalSelfAttention`, `TransformerBlock`,
  `ChatGPTMini.forward`, and `generate()` all updated). No retraining needed -- same
  weights, only the computation path changed (prefill the prompt once, then feed only
  the newest token each step instead of reprocessing the whole sequence). This is the
  standard technique used by all production LLM inference stacks.
- **Final decision: ship fp32, do NOT use int8 quantization.** Benchmark history:
  | condition | fp32 gens/sec | int8 gens/sec | speedup |
  |---|---|---|---|
  | batch=1, no KV-cache, power-save | 9.39 | 8.21 | 0.83x |
  | batch=32, no KV-cache, power-save | 7.36 | 9.98 | 1.35x |
  | batch=32, no KV-cache, normal battery | 7.74 | 6.59 | 0.85x |
  | batch=32, **WITH KV-cache**, normal battery | **36.08** | 14.10 | 0.46x |

  Without caching, fp32-vs-int8 results were noisy/inconsistent across runs (not a
  reliable signal either way). With caching added, fp32 jumped ~4.7x and pulled clearly
  ahead of int8 (which got relatively worse, since quantization's per-call overhead
  is now a bigger fraction of a much smaller per-step computation). Conclusion: int8
  dynamic quantization is not a good fit for this model+workload once KV-caching is in
  place. **The quantized files remain in the repo for reference but are not the shipped
  path.**
- **Result: 36 generations/sec, 920 tokens/sec, batch=32, 30 tokens each, fp32 CPU.**
  This comfortably satisfies the "many short comments/sec" real-time requirement.
- **Not started yet**: the keyword-injection layer (prompt-seeding with transcript
  keywords, or generate-then-filter matching) — separate code on top of the finished
  model, no architecture/training changes needed. See "Context/relevance problem"
  section above for the design (Option 2 chosen: runtime injection, not training-time
  conditioning). **This is now the main remaining piece of the whole project.**

## Repo / backup status
- Two separate concerns, handled separately:
  - **Code** (this folder, `chat_ai_model`) — meant for GitHub backup. `.gitignore`
    added, excludes `checkpoints/*.pt`, `*.safetensors`, `*.bin` (large binaries —
    GitHub has a 100MB/file limit and gets bloated/slow with model weights in git
    history anyway). `.gitkeep` files are kept so empty folders still track in git.
  - **Model weights** — NOT in GitHub, currently local-only in `checkpoints/`. Plan is
    to publish the trained weights + tokenizer + model card to Hugging Face separately
    (see below). Back up `checkpoints/` manually (external drive/cloud) in the meantime
    if you don't want to risk losing the completed training run.

## Hugging Face publishing plan (discussed, not started)
Worth doing for resume purposes once training/quantization/inference are finalized.
- Upload: model weights (safetensors preferred over raw `.pt`), tokenizer files, and
  `model/model.py` itself (fully custom architecture, not a registered HF architecture
  — people loading it need the class definition; normal for small custom/hobby models
  on HF, not a blocker).
- No GGUF needed/planned — GGUF only matters for llama.cpp/Ollama/LM Studio
  compatibility; since this is a custom architecture it was never a natural fit for
  that ecosystem anyway. Plain PyTorch + safetensors is a normal, common way to share
  small models on HF.
- Model card must credit the `lparkourer10/twitch_chat` dataset (CC-BY-SA-4.0) and
  carry a compatible license.
- Optional stretch: a Hugging Face Space with a live interactive demo (bigger resume
  value than static weights alone).

## Known environment quirks (useful if picking this up again)
- User runs Windows + PowerShell. Watch for path drift issues (has happened multiple
  times — always confirm `pwd`/prompt path before assuming a script ran in the right
  place; the two project folders are siblings, easy to `cd` into the wrong one).
- Multiple projects are registered in the MCP file tool; always confirm active project
  before writing files (has drifted mid-session before).
- `torch.load` throws a `weights_only=False` FutureWarning on every checkpoint load —
  expected/harmless for our own trusted local files, not an actual problem.

## v1 STATUS: COMPLETE (2026-07-23)
Extended training (60,000 steps total, resumed from the 30k checkpoint) + expanded
superchat data (roasts/jokes/personal questions added) + the keyword-injection bug fix
(contractions like "let's" were slipping past the stopword filter and dominating
generations) together produced a v1 the user considers good enough to ship. Sample
output at 60k steps: mostly readable, on-topic chat with real Twitch texture (correct
emote usage, @-mentions, streamer name references), some residual garbling concentrated
in the highest-chaos persona preset. Considered acceptable given real chat's inherent
noisiness — further improvement would mean raw-data cleaning, a real but
diminishing-returns next step, not required to consider this done.

## Immediate next step (updated)
Core project (data → tokenizer → model → training → inference → keyword injection →
demo) is functionally complete. Remaining work is publishing: build the Hugging Face
model card, upload weights (safetensors) + tokenizer + model/model.py, and push code to
GitHub (`.gitignore` already set up for this). See "Hugging Face publishing plan" above.
