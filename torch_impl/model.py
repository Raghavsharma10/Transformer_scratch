"""
torch_impl/model.py
Production decoder-only transformer for Python code completion.

Architecture:
  - Decoder-only (causal LM), GPT-style
  - Rotary Position Embeddings (RoPE)
  - Pre-LayerNorm (more stable than post-LN)
  - SwiGLU feed-forward (better than vanilla GeLU)
  - ~10 M parameters at default config

Default config  (smoke-test on CPU in minutes):
  vocab=8000, d=256, heads=8, layers=6, d_ff=1024  → ~10 M params

Larger config for full training (needs GPU):
  vocab=8000, d=512, heads=8, layers=8, d_ff=2048  → ~45 M params
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────

@dataclass
class TransformerConfig:
    vocab_size:  int   = 8000
    d_model:     int   = 256
    num_heads:   int   = 8
    num_layers:  int   = 6
    d_ff:        int   = 1024
    max_seq_len: int   = 512
    dropout:     float = 0.1
    pad_id:      int   = 0

    @property
    def d_head(self) -> int:
        assert self.d_model % self.num_heads == 0
        return self.d_model // self.num_heads

    def param_count(self) -> int:
        """Approximate parameter count."""
        emb   = self.vocab_size * self.d_model
        attn  = 4 * self.d_model * self.d_model          # Q K V O
        ff    = 3 * self.d_model * self.d_ff              # SwiGLU has 3 matrices
        ln    = 2 * 2 * self.d_model                      # 2 LN per layer
        layer = attn + ff + ln
        total = emb + self.num_layers * layer + self.d_model * self.vocab_size
        return total


# ──────────────────────────────────────────────
# Rotary Position Embeddings (RoPE)
# ──────────────────────────────────────────────

class RotaryEmbedding(nn.Module):
    """RoPE: rotates Q and K vectors to encode relative position."""

    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10_000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)                 # (T, dim)
        self.register_buffer("cos_cached", emb.cos()[None, None])  # (1,1,T,dim)
        self.register_buffer("sin_cached", emb.sin()[None, None])

    def forward(self, seq_len: int):
        if seq_len > self.cos_cached.shape[2]:
            self._build_cache(seq_len)
        return (
            self.cos_cached[:, :, :seq_len, :],
            self.sin_cached[:, :, :seq_len, :],
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot


# ──────────────────────────────────────────────
# Multi-head Self-Attention with RoPE + causal mask
# ──────────────────────────────────────────────

class CausalSelfAttention(nn.Module):

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.d_head    = config.d_head
        self.d_model   = config.d_model

        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.rope         = RotaryEmbedding(config.d_head, config.max_seq_len)

        # Causal mask — upper-triangular -inf
        mask = torch.triu(torch.ones(config.max_seq_len, config.max_seq_len), diagonal=1).bool()
        self.register_buffer("causal_mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, d_h = self.num_heads, self.d_head

        Q = self.q_proj(x).view(B, T, H, d_h).transpose(1, 2)  # (B,H,T,d_h)
        K = self.k_proj(x).view(B, T, H, d_h).transpose(1, 2)
        V = self.v_proj(x).view(B, T, H, d_h).transpose(1, 2)

        # Apply RoPE to Q and K
        cos, sin = self.rope(T)
        Q, K = apply_rope(Q, K, cos, sin)

        # Scaled dot-product attention with causal mask
        scale  = math.sqrt(d_h)
        scores = (Q @ K.transpose(-2, -1)) / scale          # (B,H,T,T)
        scores = scores.masked_fill(self.causal_mask[:T, :T], float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)

        ctx = (weights @ V).transpose(1, 2).contiguous().view(B, T, D)
        return self.o_proj(ctx)


# ──────────────────────────────────────────────
# SwiGLU Feed-forward
# ──────────────────────────────────────────────

class SwiGLUFFN(nn.Module):
    """
    SwiGLU: FFN(x) = (W1·x ⊙ SiLU(W3·x)) · W2
    ~15 % better than vanilla GeLU FFN at same param count.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.w1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.w2 = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.w3 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.w2(F.silu(self.w1(x)) * self.w3(x)))


# ──────────────────────────────────────────────
# Transformer block (Pre-LN)
# ──────────────────────────────────────────────

class TransformerBlock(nn.Module):

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.ln1  = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln2  = nn.LayerNorm(config.d_model)
        self.ffn  = SwiGLUFFN(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


# ──────────────────────────────────────────────
# Full Decoder-only Transformer
# ──────────────────────────────────────────────

class PythonCodeTransformer(nn.Module):

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.d_model,
                                       padding_idx=config.pad_id)
        self.emb_drop  = nn.Dropout(config.dropout)
        self.blocks    = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.ln_f      = nn.LayerNorm(config.d_model)
        self.lm_head   = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Tie token embedding weights with LM head (standard for LMs)
        self.lm_head.weight = self.token_emb.weight

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,            # (B, T)
        labels: Optional[torch.Tensor] = None,  # (B, T) for training
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns (logits, loss).
        loss is None when labels is None (inference mode).
        """
        B, T = input_ids.shape
        assert T <= self.config.max_seq_len, f"Sequence length {T} exceeds max {self.config.max_seq_len}"

        x = self.emb_drop(self.token_emb(input_ids))     # (B, T, D)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)                          # (B, T, V)

        loss = None
        if labels is not None:
            # Ignore padding tokens in loss
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                labels.view(-1),
                ignore_index=self.config.pad_id,
            )

        return logits, loss

    @torch.no_grad()
    def generate_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return logits for the last token position (for autoregressive generation)."""
        logits, _ = self.forward(input_ids)
        return logits[:, -1, :]   # (B, V)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
