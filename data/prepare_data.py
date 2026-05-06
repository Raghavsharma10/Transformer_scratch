"""
data/prepare_data.py
Download and pre-process Python code corpus for training.

Datasets used (in order of preference):
  1. CodeSearchNet  — python split  (~500 K functions, Apache-2.0)
  2. The Stack (optional, requires HF token and acceptance of terms)

Output layout:
  data/raw_python/        — one .py file per function (AST-valid only)
  data/processed/train.jsonl
  data/processed/val.jsonl
  data/processed/test.jsonl

Usage:
    pip install datasets tqdm
    python -m data.prepare_data --output_dir data --max_samples 50000
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import random
import re
from pathlib import Path
from typing import Iterator


# ──────────────────────────────────────────────
# Filters
# ──────────────────────────────────────────────

def is_valid_python(code: str) -> bool:
    """Return True iff the code parses without SyntaxError."""
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def is_quality_function(code: str, min_lines: int = 3, max_lines: int = 200) -> bool:
    """Heuristic quality filter: skip trivial stubs and monster functions."""
    lines = [l for l in code.splitlines() if l.strip()]
    if not (min_lines <= len(lines) <= max_lines):
        return False
    # Must contain at least one expression or assignment (not just `pass`)
    if all(l.strip() in ("pass", "...", "return") for l in lines[1:]):
        return False
    return True


def strip_comments_and_docstrings(code: str) -> str:
    """
    Optionally remove docstrings and inline comments to reduce noise.
    Disabled by default — kept here for experimentation.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            if isinstance(node.value.value, str):
                node.value.value = ""
    return ast.unparse(tree)


# ──────────────────────────────────────────────
# CodeSearchNet loader
# ──────────────────────────────────────────────

def load_codesearchnet(max_samples: int | None = None) -> Iterator[dict]:
    """
    Yields dicts with keys: code, docstring, repo, path.
    Requires: pip install datasets
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Run: pip install datasets")

    print("Downloading CodeSearchNet Python split …")
    ds = load_dataset("code_search_net", "python", trust_remote_code=True)

    count = 0
    for split in ("train", "validation", "test"):
        for ex in ds[split]:
            code = ex.get("whole_func_string", "")
            if not code:
                continue
            yield {
                "code":      code,
                "docstring": ex.get("func_documentation_string", ""),
                "repo":      ex.get("repo", ""),
                "path":      ex.get("path", ""),
                "split":     split,
            }
            count += 1
            if max_samples and count >= max_samples:
                return


# ──────────────────────────────────────────────
# Sliding-window context builder
# ──────────────────────────────────────────────

def build_sliding_windows(
    codes: list[str],
    tokenizer,
    context_len: int = 512,
    stride: int = 256,
) -> list[list[int]]:
    """
    Tokenise each code snippet and create overlapping windows of length context_len.
    Returns list of token-id lists, each exactly context_len long.
    """
    windows: list[list[int]] = []
    for code in codes:
        ids = tokenizer.encode(code, add_bof=True, add_eof=True)
        for start in range(0, max(1, len(ids) - context_len + 1), stride):
            chunk = ids[start: start + context_len]
            if len(chunk) < context_len:
                # Pad to context_len
                from tokenizer.bpe import PAD_ID
                chunk = chunk + [PAD_ID] * (context_len - len(chunk))
            windows.append(chunk)
    return windows


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Python training data")
    parser.add_argument("--output_dir",   default="data",
                        help="Root output directory")
    parser.add_argument("--max_samples",  type=int, default=None,
                        help="Cap total number of functions (all splits)")
    parser.add_argument("--val_ratio",    type=float, default=0.05,
                        help="Fraction of data for validation")
    parser.add_argument("--test_ratio",   type=float, default=0.05,
                        help="Fraction of data for test")
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--save_py_files", action="store_true",
                        help="Also save individual .py files to data/raw_python/")
    args = parser.parse_args()

    random.seed(args.seed)
    raw_dir  = Path(args.output_dir) / "raw_python"
    proc_dir = Path(args.output_dir) / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)

    # ── Collect functions ──
    all_records: list[dict] = []
    total_loaded  = 0
    total_kept    = 0

    for rec in load_codesearchnet(max_samples=args.max_samples):
        total_loaded += 1
        code = rec["code"]
        if not is_valid_python(code):
            continue
        if not is_quality_function(code):
            continue
        all_records.append(rec)
        total_kept += 1
        if total_loaded % 10_000 == 0:
            print(f"  Processed {total_loaded:,} functions, kept {total_kept:,} …")

    print(f"\nTotal loaded: {total_loaded:,}  |  After filtering: {total_kept:,}")

    # ── Optional: save raw .py files for tokenizer training ──
    if args.save_py_files:
        print(f"Saving .py files to {raw_dir} …")
        for i, rec in enumerate(all_records):
            fname = raw_dir / f"func_{i:07d}.py"
            with open(fname, "w", encoding="utf-8") as f:
                f.write(rec["code"])

    # ── Train / val / test split ──
    random.shuffle(all_records)
    n = len(all_records)
    n_val   = max(1, int(n * args.val_ratio))
    n_test  = max(1, int(n * args.test_ratio))
    n_train = n - n_val - n_test

    splits = {
        "train": all_records[:n_train],
        "val":   all_records[n_train: n_train + n_val],
        "test":  all_records[n_train + n_val:],
    }

    for split_name, records in splits.items():
        out_path = proc_dir / f"{split_name}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps({"code": rec["code"]}, ensure_ascii=False) + "\n")
        print(f"  {split_name}: {len(records):,} functions → {out_path}")

    print("\nData preparation complete.")
    print(f"Next step: train the tokenizer:")
    print(f"  python -m tokenizer.train_tokenizer --data_dir {raw_dir} --vocab_size 8000")


if __name__ == "__main__":
    main()
