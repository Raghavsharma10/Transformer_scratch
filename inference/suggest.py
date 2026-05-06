"""
inference/suggest.py
Top-k code suggestion engine using diverse beam search.

Strategy:
  - Run 3 independent autoregressive samples with different temperature /
    diversity settings so each beam explores a distinct region of the distribution.
  - Rank completions by (syntax validity) + (log-probability score).
  - Deduplicate by normalised edit distance.

Usage:
    from inference.suggest import CodeSuggester
    suggester = CodeSuggester("checkpoints/best.pt", "tokenizer/python_bpe.model")
    results = suggester.suggest("def fibonacci(n):\n    ", k=3)
    for r in results:
        print(r.completion)
"""

from __future__ import annotations

import ast
import math
import time
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F

from torch_impl.model import PythonCodeTransformer, TransformerConfig
from tokenizer.bpe import PythonBPETokenizer, EOF_ID, NEWLINE_ID


# ──────────────────────────────────────────────
# Result dataclass
# ──────────────────────────────────────────────

@dataclass
class Suggestion:
    completion:  str          # the generated text (continuation only, not the prefix)
    score:       float        # composite score (higher = better)
    log_prob:    float        # raw log-probability
    is_valid_py: bool         # passes ast.parse?
    latency_ms:  float        # wall-clock time to generate this suggestion


# ──────────────────────────────────────────────
# Sampling helpers
# ──────────────────────────────────────────────

def top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Nucleus filtering: zero out logits outside the top-p probability mass."""
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    # Remove tokens once cumulative prob exceeds top_p
    sorted_logits[cumulative_probs - F.softmax(sorted_logits, dim=-1) > top_p] = float("-inf")
    # Scatter back to original indexing
    logits_filtered = logits.clone()
    logits_filtered.scatter_(0, sorted_idx, sorted_logits)
    return logits_filtered


def sample_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> tuple[int, float]:
    """Sample one token; return (token_id, log_prob)."""
    if temperature == 0.0:
        token_id = int(logits.argmax())
        log_prob  = float(F.log_softmax(logits, dim=-1)[token_id])
        return token_id, log_prob

    scaled = logits / temperature
    if top_p < 1.0:
        scaled = top_p_filter(scaled, top_p)

    probs    = F.softmax(scaled, dim=-1)
    token_id = int(torch.multinomial(probs, num_samples=1))
    log_prob = float(torch.log(probs[token_id] + 1e-10))
    return token_id, log_prob


# ──────────────────────────────────────────────
# Stop conditions
# ──────────────────────────────────────────────

def _should_stop(token_id: int, generated_ids: list[int],
                 tokenizer: PythonBPETokenizer, max_tokens: int) -> bool:
    if len(generated_ids) >= max_tokens:
        return True
    if token_id == EOF_ID:
        return True
    # Stop after generating a complete top-level statement (two consecutive newlines)
    if len(generated_ids) >= 2:
        decoded = tokenizer.decode(generated_ids[-4:], skip_special=False)
        if "\n\n" in decoded:
            return True
    return False


# ──────────────────────────────────────────────
# Single-beam autoregressive sampling
# ──────────────────────────────────────────────

@torch.no_grad()
def _sample_beam(
    model: PythonCodeTransformer,
    input_ids: torch.Tensor,          # (1, T) on device
    tokenizer: PythonBPETokenizer,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: torch.device,
) -> tuple[list[int], float]:
    """
    Generate one completion autoregressively.
    Returns (generated_token_ids, cumulative_log_prob).
    """
    generated: list[int] = []
    cum_log_prob = 0.0

    context = input_ids.clone()
    max_ctx = model.config.max_seq_len

    for _ in range(max_new_tokens):
        # Truncate context to max_seq_len
        ctx = context[:, -max_ctx:]
        logits = model.generate_logits(ctx)    # (1, V)
        tok_id, lp = sample_token(logits[0], temperature=temperature, top_p=top_p)
        cum_log_prob += lp
        generated.append(tok_id)

        if _should_stop(tok_id, generated, tokenizer, max_new_tokens):
            break

        context = torch.cat([context, torch.tensor([[tok_id]], device=device)], dim=1)

    return generated, cum_log_prob


# ──────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────

def _is_valid_python(prefix: str, completion: str) -> bool:
    try:
        ast.parse(prefix + completion)
        return True
    except SyntaxError:
        pass
    # Also accept if the completion alone is valid (might be a body fragment)
    try:
        ast.parse(completion)
        return True
    except SyntaxError:
        return False


def _edit_distance(a: str, b: str) -> int:
    """Simple O(n*m) Levenshtein for deduplication."""
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def _deduplicate(suggestions: list[Suggestion], threshold: float = 0.3) -> list[Suggestion]:
    """Remove near-duplicate completions (edit distance / max_len < threshold)."""
    kept: list[Suggestion] = []
    for s in suggestions:
        is_dup = False
        for k in kept:
            max_len = max(len(s.completion), len(k.completion), 1)
            if _edit_distance(s.completion[:100], k.completion[:100]) / max_len < threshold:
                is_dup = True
                break
        if not is_dup:
            kept.append(s)
    return kept


# ──────────────────────────────────────────────
# Scoring
# ──────────────────────────────────────────────

def _composite_score(log_prob: float, is_valid: bool, completion: str) -> float:
    """
    Composite ranking score.
    Components:
      - Normalised log-probability (length-normalised so longer ≠ better)
      - Syntax validity bonus
      - Mild length preference for multi-line completions
    """
    lines = [l for l in completion.split("\n") if l.strip()]
    length_bonus  = min(len(lines), 5) / 5 * 0.2
    validity_bonus = 1.0 if is_valid else 0.0
    tokens = max(len(completion.split()), 1)
    norm_lp = log_prob / tokens
    return norm_lp + validity_bonus + length_bonus


# ──────────────────────────────────────────────
# Main suggester class
# ──────────────────────────────────────────────

# Beam configurations — intentionally different to maximise diversity
_BEAM_CONFIGS = [
    # (temperature, top_p, label)
    (0.6, 1.00, "confident"),      # beam 1: peaked, deterministic-leaning
    (0.9, 0.95, "diverse"),        # beam 2: moderate diversity
    (1.1, 0.90, "exploratory"),    # beam 3: high entropy, creative
]


class CodeSuggester:
    """
    High-level interface for top-k code completion suggestions.

    Args:
        checkpoint_path : Path to a .pt file saved by torch_impl/train.py
        tokenizer_path  : Path to the .model file from tokenizer/train_tokenizer.py
        device          : 'cuda', 'cpu', or 'mps'  (auto-detected if None)
    """

    def __init__(
        self,
        checkpoint_path: str,
        tokenizer_path: str,
        device: Optional[str] = None,
    ) -> None:
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)

        # ── Load tokenizer ──
        self.tokenizer = PythonBPETokenizer(tokenizer_path)

        # ── Load model ──
        ckpt   = torch.load(checkpoint_path, map_location="cpu")
        config = TransformerConfig(**ckpt["config"])
        self.model = PythonCodeTransformer(config)
        self.model.load_state_dict(ckpt["model"])
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded from {checkpoint_path}  |  device={device}  "
              f"|  params={self.model.param_count():,}")

    def suggest(
        self,
        prefix: str,
        k: int = 3,
        max_new_tokens: int = 64,
        deduplicate: bool = True,
    ) -> List[Suggestion]:
        """
        Generate up to k code completion suggestions for the given prefix.

        Args:
            prefix         : Code typed so far (the prompt)
            k              : Number of suggestions to return (max 3)
            max_new_tokens : Maximum tokens to generate per beam
            deduplicate    : Remove near-duplicate completions

        Returns:
            List of Suggestion objects, sorted by score descending.
        """
        t_total = time.perf_counter()

        input_ids = self.tokenizer.encode(prefix, add_bof=True, add_eof=False)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        suggestions: list[Suggestion] = []
        configs = _BEAM_CONFIGS[:max(k, 3)]   # always run 3 beams, trim after ranking

        for temp, top_p, _ in configs:
            t0 = time.perf_counter()
            gen_ids, log_prob = _sample_beam(
                self.model, input_tensor, self.tokenizer,
                max_new_tokens=max_new_tokens,
                temperature=temp, top_p=top_p,
                device=self.device,
            )
            completion = self.tokenizer.decode(gen_ids, skip_special=True)
            is_valid   = _is_valid_python(prefix, completion)
            score      = _composite_score(log_prob, is_valid, completion)
            latency    = (time.perf_counter() - t0) * 1000

            suggestions.append(Suggestion(
                completion=completion,
                score=score,
                log_prob=log_prob,
                is_valid_py=is_valid,
                latency_ms=latency,
            ))

        # Sort by score (best first)
        suggestions.sort(key=lambda s: s.score, reverse=True)

        if deduplicate:
            suggestions = _deduplicate(suggestions)

        return suggestions[:k]
