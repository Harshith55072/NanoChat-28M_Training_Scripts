"""
Model architecture: small GPT-style decoder-only transformer.

Target: ~20-30M params, fast on CPU after quantization.
Chosen config lands at ~27.7M params -- see count_params() at the bottom, or run:
    python model/model.py
to print the exact param count for a sanity check.

Design choices:
- Pre-norm transformer blocks (LayerNorm before attention/FFN, not after) -- more stable
  training for small models, standard in modern small LMs (GPT-NeoX, LLaMA style).
- Learned positional embeddings (not rotary) -- simpler to implement correctly, and chat
  comments are short (max_seq_len=128 is generous), so no need for length-extrapolation
  tricks that rotary/ALiBi exist to solve.
- Weight-tied input/output embeddings -- saves ~3M params, standard practice for small LMs.
- Causal self-attention (each token can only see previous tokens) -- required for
  autoregressive generation (predicting next token).
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 8000       # must match tokenizer vocab_size
    d_model: int = 448           # hidden dimension
    n_layer: int = 10            # number of transformer blocks
    n_head: int = 8              # attention heads (head_dim = d_model / n_head = 56)
    d_ff: int = 1792             # feedforward inner dimension (4x d_model, standard)
    max_seq_len: int = 128       # max tokens per sequence (chat comments are short)
    dropout: float = 0.1
    pad_token_id: int = 0        # index of <pad> in the tokenizer vocab


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.d_model // cfg.n_head

        self.qkv_proj = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

        # causal mask for the no-cache path (training / full-sequence forward), where
        # query length == key length == T. The cached-decode path builds its own mask
        # on the fly instead, since query/key lengths differ there (see forward()).
        mask = torch.tril(torch.ones(cfg.max_seq_len, cfg.max_seq_len))
        self.register_buffer("causal_mask", mask.view(1, 1, cfg.max_seq_len, cfg.max_seq_len))

    def forward(self, x, past_kv=None, use_cache=False):
        """
        x: (B, T_new, C) -- T_new is the full sequence on the first/no-cache call,
           or just 1 new token on subsequent cached decode steps.
        past_kv: optional (past_k, past_v), each (B, n_head, T_past, head_dim), from
           a previous call. If given, this call's new k/v are appended to them.
        """
        B, T_new, C = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(C, dim=2)

        q = q.view(B, T_new, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T_new, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T_new, self.n_head, self.head_dim).transpose(1, 2)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        present_kv = (k, v) if use_cache else None
        T_total = k.size(2)
        past_len = T_total - T_new

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))

        if past_kv is None and past_len == 0:
            # standard full-sequence causal mask (training, or first prefill call)
            mask = self.causal_mask[:, :, :T_new, :T_total]
        else:
            # cached decode: new query positions are [past_len, past_len+T_new), and
            # each may attend to all key positions up to and including itself.
            q_pos = torch.arange(past_len, past_len + T_new, device=x.device).view(1, 1, T_new, 1)
            k_pos = torch.arange(T_total, device=x.device).view(1, 1, 1, T_total)
            mask = (k_pos <= q_pos).float()

        att = att.masked_fill(mask == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        out = att @ v
        out = out.transpose(1, 2).contiguous().view(B, T_new, C)
        out = self.resid_dropout(self.out_proj(out))
        return out, present_kv


class FeedForward(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.fc1 = nn.Linear(cfg.d_model, cfg.d_ff)
        self.fc2 = nn.Linear(cfg.d_ff, cfg.d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.dropout(self.fc2(self.act(self.fc1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ffn = FeedForward(cfg)

    def forward(self, x, past_kv=None, use_cache=False):
        attn_out, present_kv = self.attn(self.ln1(x), past_kv=past_kv, use_cache=use_cache)
        x = x + attn_out             # pre-norm + residual
        x = x + self.ffn(self.ln2(x))  # pre-norm + residual
        return x, present_kv


class ChatGPTMini(nn.Module):
    """Small decoder-only transformer LM for the chat/superchat generator."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_token_id)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)

        # output head, weight-tied to token_emb (saves ~3M params, standard practice)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, targets=None, past_kv=None, use_cache=False):
        B, T = input_ids.shape
        past_len = past_kv[0][0].size(2) if past_kv is not None else 0
        assert past_len + T <= self.cfg.max_seq_len, (
            f"sequence length {past_len + T} exceeds max_seq_len {self.cfg.max_seq_len}"
        )

        pos = torch.arange(past_len, past_len + T, device=input_ids.device).unsqueeze(0)
        x = self.token_emb(input_ids) + self.pos_emb(pos)
        x = self.dropout(x)

        present_kvs = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            layer_past = past_kv[i] if past_kv is not None else None
            x, present_kv = block(x, past_kv=layer_past, use_cache=use_cache)
            if use_cache:
                present_kvs.append(present_kv)
        x = self.ln_f(x)

        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=self.cfg.pad_token_id,
            )

        if use_cache:
            return logits, loss, present_kvs
        return logits, loss

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=40, temperature=0.9, top_k=40, top_p=0.9, eos_token_id=None):
        """Autoregressive sampling with KV-caching. input_ids: (B, T) prompt tokens.

        Speed note: without caching, every new token re-runs the forward pass over the
        ENTIRE sequence so far (cost grows quadratically with length). With caching, the
        prompt is processed once ("prefill"), then each new token only needs a forward
        pass over that single new token, reusing cached keys/values from every previous
        step (cost grows linearly). This is the standard technique used by every
        production LLM inference stack.
        """
        self.eval()
        B = input_ids.size(0)

        # prefill: process the whole prompt at once, building the initial cache
        logits, _, past_kv = self(input_ids, use_cache=True)
        next_logits = logits[:, -1, :]

        generated = input_ids
        finished = torch.zeros(B, dtype=torch.bool, device=input_ids.device)

        for _ in range(max_new_tokens):
            logits_t = next_logits / max(temperature, 1e-5)

            if top_k is not None:
                v, _ = torch.topk(logits_t, min(top_k, logits_t.size(-1)))
                logits_t[logits_t < v[:, [-1]]] = float("-inf")

            if top_p is not None:
                sorted_logits, sorted_idx = torch.sort(logits_t, descending=True)
                probs = F.softmax(sorted_logits, dim=-1)
                cumprobs = torch.cumsum(probs, dim=-1)
                remove = cumprobs > top_p
                remove[:, 1:] = remove[:, :-1].clone()
                remove[:, 0] = False
                sorted_logits[remove] = float("-inf")
                logits_t = torch.full_like(logits_t, float("-inf")).scatter(1, sorted_idx, sorted_logits)

            probs = F.softmax(logits_t, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)

            generated = torch.cat([generated, next_token], dim=1)

            if eos_token_id is not None:
                finished = finished | (next_token.squeeze(1) == eos_token_id)
                if finished.all():
                    break

            if generated.size(1) >= self.cfg.max_seq_len:
                break

            # only feed the single new token -- past_kv already holds everything before it
            logits, _, past_kv = self(next_token, past_kv=past_kv, use_cache=True)
            next_logits = logits[:, -1, :]

        return generated


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    cfg = ModelConfig()
    model = ChatGPTMini(cfg)
    n_params = count_params(model)
    print(f"ChatGPTMini config: {cfg}")
    print(f"Total parameters: {n_params:,} ({n_params/1e6:.2f}M)")

    # quick forward-pass sanity check with random input
    dummy = torch.randint(0, cfg.vocab_size, (2, 20))
    logits, loss = model(dummy, targets=dummy)
    print(f"Sanity forward pass -> logits shape: {tuple(logits.shape)}, loss: {loss.item():.4f}")
