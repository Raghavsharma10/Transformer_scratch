"""
data/dataset.py
PyTorch Dataset for Python code completion training.
Supports both JSONL files (from prepare_data.py) and raw .py file directories.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader


class PythonCodeDataset(Dataset):
    """
    Sliding-window token dataset for causal language modelling.

    Each item is (input_ids, target_ids) where target_ids = input_ids shifted
    left by one position (standard next-token prediction objective).

    Args:
        token_windows : list of token-id lists, each of length `context_len`
        context_len   : window length (default 512)
    """

    def __init__(self, token_windows: List[List[int]], context_len: int = 512):
        self.windows     = token_windows
        self.context_len = context_len

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ids = torch.tensor(self.windows[idx], dtype=torch.long)
        # input  = all tokens except last
        # target = all tokens except first (shift-by-one)
        input_ids  = ids[:-1]
        target_ids = ids[1:]
        return input_ids, target_ids


# ──────────────────────────────────────────────
# Factory helpers
# ──────────────────────────────────────────────

def load_jsonl_dataset(
    jsonl_path: str,
    tokenizer,
    context_len: int = 512,
    stride: int = 256,
    max_samples: Optional[int] = None,
) -> PythonCodeDataset:
    """Build a dataset from a .jsonl file produced by prepare_data.py."""
    codes: List[str] = []
    with open(jsonl_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            obj = json.loads(line)
            codes.append(obj["code"])

    windows = _build_windows(codes, tokenizer, context_len, stride)
    return PythonCodeDataset(windows, context_len)


def load_pyfiles_dataset(
    directory: str,
    tokenizer,
    context_len: int = 512,
    stride: int = 256,
    max_files: Optional[int] = None,
) -> PythonCodeDataset:
    """Build a dataset by scanning a directory for .py files."""
    py_files = sorted(Path(directory).rglob("*.py"))
    if max_files:
        py_files = py_files[:max_files]

    codes: List[str] = []
    for p in py_files:
        try:
            codes.append(p.read_text(encoding="utf-8", errors="ignore"))
        except OSError:
            continue

    windows = _build_windows(codes, tokenizer, context_len, stride)
    return PythonCodeDataset(windows, context_len)


def _build_windows(
    codes: List[str],
    tokenizer,
    context_len: int,
    stride: int,
) -> List[List[int]]:
    from tokenizer.bpe import PAD_ID
    windows: List[List[int]] = []
    for code in codes:
        ids = tokenizer.encode(code, add_bof=True, add_eof=True)
        # +1 because the dataset shifts by 1 (need context_len + 1 tokens)
        window_size = context_len + 1
        for start in range(0, max(1, len(ids) - window_size + 1), stride):
            chunk = ids[start: start + window_size]
            if len(chunk) < window_size:
                chunk = chunk + [PAD_ID] * (window_size - len(chunk))
            windows.append(chunk)
    return windows


# ──────────────────────────────────────────────
# DataLoader factory
# ──────────────────────────────────────────────

def make_dataloaders(
    train_jsonl: str,
    val_jsonl: str,
    tokenizer,
    context_len: int = 512,
    stride: int = 256,
    batch_size: int = 16,
    num_workers: int = 4,
    max_train_samples: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    train_ds = load_jsonl_dataset(train_jsonl, tokenizer, context_len, stride,
                                   max_samples=max_train_samples)
    val_ds   = load_jsonl_dataset(val_jsonl,   tokenizer, context_len, stride)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    print(f"Train windows: {len(train_ds):,}  |  Val windows: {len(val_ds):,}")
    return train_loader, val_loader
