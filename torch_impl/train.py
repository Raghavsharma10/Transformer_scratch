"""
torch_impl/train.py
Training script for the Python code completion transformer.

Features:
  - AdamW optimiser with weight decay
  - Cosine LR schedule with linear warmup
  - Gradient clipping
  - Checkpoint save / resume
  - Optional W&B logging
  - Mixed precision (fp16/bf16) when GPU available

Usage (quick smoke test on CPU):
    python -m torch_impl.train \
        --train_jsonl data/processed/train.jsonl \
        --val_jsonl   data/processed/val.jsonl \
        --tokenizer   tokenizer/python_bpe.model \
        --output_dir  checkpoints \
        --max_steps   500 \
        --batch_size  4

Full GPU training:
    python -m torch_impl.train \
        --train_jsonl data/processed/train.jsonl \
        --val_jsonl   data/processed/val.jsonl \
        --tokenizer   tokenizer/python_bpe.model \
        --output_dir  checkpoints \
        --d_model 256 --num_layers 6 --num_heads 8 --d_ff 1024 \
        --batch_size 32 --max_steps 100000
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torch_impl.model import PythonCodeTransformer, TransformerConfig
from tokenizer.bpe import PythonBPETokenizer
from data.dataset import make_dataloaders


# ──────────────────────────────────────────────
# LR schedule
# ──────────────────────────────────────────────

def cosine_lr_with_warmup(
    step: int,
    warmup_steps: int,
    max_steps: int,
    max_lr: float,
    min_lr: float,
) -> float:
    if step < warmup_steps:
        return max_lr * step / max(1, warmup_steps)
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + cosine * (max_lr - min_lr)


def set_lr(optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


# ──────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────

@torch.no_grad()
def evaluate(model: PythonCodeTransformer, loader: DataLoader, device: torch.device,
             max_batches: int = 50) -> float:
    model.eval()
    total_loss, n = 0.0, 0
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        _, loss = model(x, labels=y)
        total_loss += loss.item()
        n += 1
    model.train()
    return total_loss / max(n, 1)


# ──────────────────────────────────────────────
# Checkpoint helpers
# ──────────────────────────────────────────────

def save_checkpoint(model, optimizer, step: int, val_loss: float, output_dir: str,
                    config: TransformerConfig) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ckpt = {
        "step":      step,
        "val_loss":  val_loss,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config":    vars(config),
    }
    path = os.path.join(output_dir, f"ckpt_step{step:07d}.pt")
    torch.save(ckpt, path)
    # Also overwrite the "best" symlink / file
    best_path = os.path.join(output_dir, "best.pt")
    torch.save(ckpt, best_path)
    print(f"  [ckpt] Saved → {path}")


def load_checkpoint(path: str, model, optimizer=None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    print(f"Resumed from {path}  (step {ckpt['step']}, val_loss {ckpt['val_loss']:.4f})")
    return ckpt["step"]


# ──────────────────────────────────────────────
# Main training loop
# ──────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    # ── Device ──
    if torch.cuda.is_available():
        device = torch.device("cuda")
        dtype  = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        dtype  = torch.float32
    else:
        device = torch.device("cpu")
        dtype  = torch.float32
    print(f"Training on {device} ({dtype})")

    # ── Tokenizer ──
    tokenizer = PythonBPETokenizer(args.tokenizer)

    # ── Model config ──
    config = TransformerConfig(
        vocab_size  = len(tokenizer),
        d_model     = args.d_model,
        num_heads   = args.num_heads,
        num_layers  = args.num_layers,
        d_ff        = args.d_ff,
        max_seq_len = args.context_len,
        dropout     = args.dropout,
        pad_id      = 0,
    )
    print(f"Model config: {config}")

    model = PythonCodeTransformer(config).to(device)
    print(f"Parameters: {model.param_count():,}")

    # ── Optimiser ──
    # Separate weight-decayed and non-decayed parameters
    decay_params    = [p for n, p in model.named_parameters()
                       if p.requires_grad and p.dim() >= 2]
    no_decay_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params,    "weight_decay": args.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=args.max_lr, betas=(0.9, 0.95), eps=1e-8,
    )

    # ── Data ──
    train_loader, val_loader = make_dataloaders(
        args.train_jsonl, args.val_jsonl, tokenizer,
        context_len=args.context_len,
        stride=args.stride,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # ── Optional W&B ──
    use_wandb = False
    if args.wandb_project:
        try:
            import wandb
            wandb.init(project=args.wandb_project, config=vars(args))
            use_wandb = True
        except ImportError:
            print("[warn] wandb not installed, skipping logging")

    # ── Resume ──
    global_step = 0
    best_val    = float("inf")
    if args.resume:
        global_step = load_checkpoint(args.resume, model, optimizer)

    # ── AMP scaler ──
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    model.train()
    train_iter = iter(train_loader)
    t0 = time.time()

    print(f"\nStarting training for {args.max_steps} steps …\n")

    for step in range(global_step, args.max_steps):
        # ── LR schedule ──
        lr = cosine_lr_with_warmup(step, args.warmup_steps, args.max_steps,
                                    args.max_lr, args.min_lr)
        set_lr(optimizer, lr)

        # ── Fetch batch (cycle iterator) ──
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)
        x, y = x.to(device), y.to(device)

        # ── Forward / backward ──
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=dtype,
                             enabled=(device.type != "cpu")):
            _, loss = model(x, labels=y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        # ── Logging ──
        if step % args.log_every == 0:
            dt = (time.time() - t0) / max(step - global_step + 1, 1)
            eta_h = (args.max_steps - step) * dt / 3600
            print(f"  step {step:6d} | loss {loss.item():.4f} | lr {lr:.2e} "
                  f"| {dt*1000:.0f}ms/step | ETA {eta_h:.1f}h")
            if use_wandb:
                wandb.log({"train/loss": loss.item(), "train/lr": lr}, step=step)

        # ── Validation + checkpoint ──
        if step % args.eval_every == 0 and step > 0:
            val_loss = evaluate(model, val_loader, device)
            print(f"  ── val_loss: {val_loss:.4f}")
            if use_wandb:
                wandb.log({"val/loss": val_loss}, step=step)
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(model, optimizer, step, val_loss,
                                 args.output_dir, config)

    # Final checkpoint
    val_loss = evaluate(model, val_loader, device)
    save_checkpoint(model, optimizer, args.max_steps, val_loss,
                     args.output_dir, config)
    print(f"\nTraining complete. Best val loss: {best_val:.4f}")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Train Python code completion transformer")
    # Data
    p.add_argument("--train_jsonl",  default="data/processed/train.jsonl")
    p.add_argument("--val_jsonl",    default="data/processed/val.jsonl")
    p.add_argument("--tokenizer",    default="tokenizer/python_bpe.model")
    p.add_argument("--context_len",  type=int, default=512)
    p.add_argument("--stride",       type=int, default=256)
    p.add_argument("--num_workers",  type=int, default=4)
    # Model
    p.add_argument("--d_model",      type=int, default=256)
    p.add_argument("--num_heads",    type=int, default=8)
    p.add_argument("--num_layers",   type=int, default=6)
    p.add_argument("--d_ff",         type=int, default=1024)
    p.add_argument("--dropout",      type=float, default=0.1)
    # Training
    p.add_argument("--batch_size",   type=int, default=16)
    p.add_argument("--max_steps",    type=int, default=100_000)
    p.add_argument("--warmup_steps", type=int, default=2_000)
    p.add_argument("--max_lr",       type=float, default=3e-4)
    p.add_argument("--min_lr",       type=float, default=3e-5)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--grad_clip",    type=float, default=1.0)
    # Logging / checkpoints
    p.add_argument("--output_dir",   default="checkpoints")
    p.add_argument("--log_every",    type=int, default=50)
    p.add_argument("--eval_every",   type=int, default=500)
    p.add_argument("--resume",       default=None, help="Path to checkpoint to resume from")
    p.add_argument("--wandb_project",default=None)

    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
