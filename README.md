# NanoChat-28M

A ~27.8M parameter GPT-style language model, trained fully from scratch on a single
consumer laptop GPU (RTX 4050, 6GB VRAM), that generates simulated live-stream chat --
both fast, low-effort "hype chat" and rarer, longer "superchat"-style messages. Built as
a lightweight, fast-on-CPU audience-simulation layer for AI VTuber / livestream setups.

**Trained model, weights, and usage instructions:** [Hugging Face model page](#) <!-- update this link after uploading -->

## What this is

Live-stream chat has a very specific texture: short, bursty, high-volume, low
grammatical complexity, and occasionally interrupted by longer paid "superchat"
messages that the streamer is expected to acknowledge. This project trains a small
transformer entirely from scratch to mimic that texture, rather than fine-tuning an
existing large model -- chosen deliberately, since the target domain is narrow enough
that a small model trained on-topic outperforms a much larger general-purpose model for
this specific use case, at a fraction of the resource cost.

## Pipeline overview

1. **Data** -- real public Twitch chat logs ([`lparkourer10/twitch_chat`](https://huggingface.co/datasets/lparkourer10/twitch_chat),
   CC-BY-SA-4.0) for the `chat` class; hand-written + template-generated examples for
   the `superchat` class, since no public superchat dataset exists.
2. **Tokenizer** -- byte-level BPE, vocab size 8,000, trained on the combined corpus.
3. **Model** -- custom GPT-style decoder-only transformer (`model/model.py`), 10 layers,
   448 hidden dim, 8 heads, ~27.8M params, weight-tied embeddings, KV-cache-enabled
   generation.
4. **Training** -- `training/train.py`: AdamW + cosine LR schedule, mixed precision,
   class-imbalance-aware sampling (superchat is a tiny fraction of raw examples but is
   sampled at a fixed rate regardless, not duplicated into the dataset).
5. **Inference** -- `inference/`: fp32 + KV-caching turned out to beat int8 dynamic
   quantization once caching was added (quantization overhead dominates at this small a
   scale/step size -- see `project.md` for the full benchmark history). ~36
   generations/sec, ~920 tokens/sec at batch_size=32 on CPU.
6. **Relevance/keyword injection** -- `inference/keyword_utils.py`: the model itself has
   no context input; topical relevance to a live stream is faked at generation time by
   occasionally seeding the prompt with a keyword extracted from recent transcript text.
7. **Demo** -- `inference/interface.py`: interactive terminal demo, type a line as the
   "streamer" and watch simulated chat react.

## Repo layout

```
model/            model architecture (ChatGPTMini, ModelConfig)
training/         training script (train.py)
inference/         quantization, benchmarking, keyword injection, terminal demo
checkpoints/       trained weights (gitignored -- large binaries, not committed)
huggingface_release/  self-contained HF upload package (gitignored, see its own README)
project.md         full internal build log / design decisions / status
```

Note: the training/preprocessing data pipeline lives in a sibling project,
`NanoChat-28M_model_data` (not included in this repo -- data collection, cleaning, and
tokenizer training happen there; see `project.md` for details).

## Running it yourself

```powershell
pip install torch tokenizers safetensors

# train from scratch (needs the sibling data project set up first, see project.md)
python training/train.py

# quantize / benchmark
python inference/quantize.py
python inference/benchmark.py

# interactive demo
python inference/interface.py
```

## Design notes worth knowing

- **Trained from scratch, not fine-tuned** -- deliberate choice, see `project.md` for
  the reasoning (narrow domain + resume value).
- **Quantization was tested and rejected** -- int8 dynamic quantization looked
  inconsistent/unhelpful in early tests and became actively counterproductive once
  KV-caching was added (per-call quantization overhead dominates at this scale once
  each decode step is already cheap). fp32 is the shipped path.
- **Superchat class imbalance** (~700 unique examples vs. millions of chat rows) is
  handled via weighted sampling during training, not data duplication.
- Full build log, every design decision, and all benchmark numbers are in `project.md`.

## License

Code: MIT (adjust if you'd prefer otherwise). Model weights inherit CC-BY-SA-4.0 from
the primary training dataset -- see `huggingface_release/README.md` for the full model
card and license details.
