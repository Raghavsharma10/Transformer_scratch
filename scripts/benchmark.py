"""
scripts/benchmark.py
Evaluate the trained model on held-out test data.

Metrics:
  - Cross-entropy loss
  - Perplexity
  - Next-token accuracy (exact match of argmax)
  - Suggestion acceptance rate simulation (top-3 oracle accuracy)
  - Latency per suggestion (p50, p95, p99)

Usage:
    python -m scripts.benchmark \
        --checkpoint checkpoints/best.pt \
        --tokenizer  tokenizer/python_bpe.model \
        --test_jsonl data/processed/test.jsonl \
        --num_samples 200
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
import numpy as np

from torch_impl.model import PythonCodeTransformer, TransformerConfig
from tokenizer.bpe import PythonBPETokenizer, PAD_ID
from inference.suggest import CodeSuggester


# ──────────────────────────────────────────────
# Loss / perplexity
# ──────────────────────────────────────────────

@torch.no_grad()
def compute_perplexity(
    model: PythonCodeTransformer,
    tokenizer: PythonBPETokenizer,
    codes: List[str],
    context_len: int = 512,
    device: torch.device = torch.device("cpu"),
) -> dict:
    model.eval()
    total_loss, total_tokens = 0.0, 0
    correct_top1 = 0

    for code in codes:
        ids = tokenizer.encode(code, add_bof=True, add_eof=True)
        if len(ids) < 4:
            continue

        # Slide through the code in context-length windows
        window = ids[: context_len + 1]
        if len(window) < 4:
            continue
        pad_len = (context_len + 1) - len(window)
        window  = window + [PAD_ID] * pad_len
        x = torch.tensor([window[:-1]], dtype=torch.long, device=device)
        y = torch.tensor([window[1:]],  dtype=torch.long, device=device)

        logits, loss = model(x, labels=y)
        total_loss   += loss.item() * (len(window) - 1)
        total_tokens += len(window) - 1

        # Top-1 accuracy
        preds        = logits[0].argmax(dim=-1)
        mask         = y[0] != PAD_ID
        correct_top1 += int((preds == y[0]).masked_select(mask).sum())

    avg_loss    = total_loss / max(total_tokens, 1)
    perplexity  = float(np.exp(avg_loss))
    accuracy    = correct_top1 / max(total_tokens, 1)

    return {
        "avg_loss":   round(avg_loss, 4),
        "perplexity": round(perplexity, 2),
        "top1_acc":   round(accuracy, 4),
        "total_tokens": total_tokens,
    }


# ──────────────────────────────────────────────
# Latency
# ──────────────────────────────────────────────

def measure_latency(
    suggester: CodeSuggester,
    prompts: List[str],
    k: int = 3,
    max_tokens: int = 32,
) -> dict:
    latencies = []
    for prompt in prompts:
        t0 = time.perf_counter()
        suggester.suggest(prompt, k=k, max_new_tokens=max_tokens)
        latencies.append((time.perf_counter() - t0) * 1000)

    latencies = sorted(latencies)
    return {
        "p50_ms":  round(np.percentile(latencies, 50), 1),
        "p95_ms":  round(np.percentile(latencies, 95), 1),
        "p99_ms":  round(np.percentile(latencies, 99), 1),
        "mean_ms": round(float(np.mean(latencies)), 1),
        "n":       len(latencies),
    }


# ──────────────────────────────────────────────
# Oracle acceptance rate
# ──────────────────────────────────────────────

def oracle_acceptance_rate(
    suggester: CodeSuggester,
    test_pairs: List[tuple[str, str]],   # (prefix, expected_continuation)
    k: int = 3,
    match_chars: int = 20,               # first N characters need to match
) -> dict:
    """
    Simulate user accepting a suggestion if any top-k suggestion starts with
    the first `match_chars` characters of the ground-truth continuation.
    """
    hits_k1, hits_k3 = 0, 0
    for prefix, expected in test_pairs:
        expected_start = expected.strip()[:match_chars]
        results = suggester.suggest(prefix, k=k, max_new_tokens=64)
        completions = [r.completion.strip() for r in results]

        if completions and completions[0].startswith(expected_start):
            hits_k1 += 1
        if any(c.startswith(expected_start) for c in completions):
            hits_k3 += 1

    n = len(test_pairs)
    return {
        "oracle_top1": round(hits_k1 / max(n, 1), 4),
        "oracle_top3": round(hits_k3 / max(n, 1), 4),
        "n":           n,
    }


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark code suggestion model")
    parser.add_argument("--checkpoint",  required=True)
    parser.add_argument("--tokenizer",   required=True)
    parser.add_argument("--test_jsonl",  required=True)
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--context_len", type=int, default=512)
    parser.add_argument("--device",      default=None)
    args = parser.parse_args()

    # ── Load data ──
    codes = []
    with open(args.test_jsonl, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= args.num_samples:
                break
            codes.append(json.loads(line)["code"])
    print(f"Loaded {len(codes)} test functions.")

    # ── Load model directly for perplexity ──
    tokenizer = PythonBPETokenizer(args.tokenizer)
    ckpt      = torch.load(args.checkpoint, map_location="cpu")
    config    = TransformerConfig(**ckpt["config"])
    model     = PythonCodeTransformer(config)
    model.load_state_dict(ckpt["model"])
    device    = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)

    # ── Perplexity ──
    print("\nComputing perplexity …")
    ppl_metrics = compute_perplexity(model, tokenizer, codes, args.context_len, device)
    print(f"  Loss:       {ppl_metrics['avg_loss']}")
    print(f"  Perplexity: {ppl_metrics['perplexity']}")
    print(f"  Top-1 acc:  {ppl_metrics['top1_acc']:.1%}")

    # ── Latency ──
    print("\nMeasuring suggestion latency …")
    suggester = CodeSuggester(args.checkpoint, args.tokenizer, device=args.device)
    # Use first 100 tokens of each code as prompts
    prompts = []
    for code in codes[:50]:
        ids    = tokenizer.encode(code)
        prefix = tokenizer.decode(ids[:80], skip_special=True)
        prompts.append(prefix)
    lat_metrics = measure_latency(suggester, prompts)
    print(f"  p50:  {lat_metrics['p50_ms']}ms")
    print(f"  p95:  {lat_metrics['p95_ms']}ms")
    print(f"  p99:  {lat_metrics['p99_ms']}ms")
    print(f"  mean: {lat_metrics['mean_ms']}ms")

    # ── Oracle ──
    print("\nComputing oracle acceptance rate …")
    test_pairs = []
    for code in codes[:100]:
        ids    = tokenizer.encode(code)
        split  = len(ids) // 2
        prefix = tokenizer.decode(ids[:split], skip_special=True)
        cont   = tokenizer.decode(ids[split:], skip_special=True)
        test_pairs.append((prefix, cont))
    oracle_metrics = oracle_acceptance_rate(suggester, test_pairs)
    print(f"  Oracle top-1: {oracle_metrics['oracle_top1']:.1%}")
    print(f"  Oracle top-3: {oracle_metrics['oracle_top3']:.1%}")

    # ── Summary ──
    print("\n" + "=" * 50)
    print("BENCHMARK SUMMARY")
    print("=" * 50)
    for k, v in {**ppl_metrics, **lat_metrics, **oracle_metrics}.items():
        print(f"  {k:<20} {v}")

    # Save as JSON
    out_path = "benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump({**ppl_metrics, **lat_metrics, **oracle_metrics}, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
