"""
tests/test_smoke.py
Fast smoke tests that run on CPU without any external data or trained weights.
Validates that all components are importable and produce correct shapes.

Run: python -m pytest tests/ -v
  or: python tests/test_smoke.py
"""

import ast
import sys
import os

# Make sure repo root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch


# ──────────────────────────────────────────────
# Core numpy transformer
# ──────────────────────────────────────────────

def test_core_forward():
    from core.transformer import TransformerParams, forward, cross_entropy_loss

    params = TransformerParams(vocab_size=50, d_model=32, num_heads=4,
                               num_layers=2, d_ff=64)
    token_ids = np.array([[1, 2, 3, 4, 5]])
    logits    = forward(params, token_ids)
    assert logits.shape == (1, 5, 50), f"Bad shape: {logits.shape}"

    targets = np.array([[2, 3, 4, 5, 0]])
    loss    = cross_entropy_loss(logits, targets)
    assert isinstance(loss, float) and loss > 0
    print(f"  [PASS] core.transformer  logits={logits.shape}  loss={loss:.4f}")


def test_causal_mask():
    from core.transformer import causal_mask
    mask = causal_mask(4)
    assert mask[0, 0] == 0.0    # self-attention allowed
    assert mask[0, 1] < -1e8    # future token blocked
    assert mask[3, 0] == 0.0    # past token allowed
    print("  [PASS] core.transformer causal mask")


# ──────────────────────────────────────────────
# Tokenizer (character-level fallback — no sentencepiece needed)
# ──────────────────────────────────────────────

def test_char_tokenizer():
    from tokenizer.bpe import CharTokenizer
    tok = CharTokenizer()
    corpus = "def foo():\n    return 42\n"
    tok.build_vocab(corpus)

    ids     = tok.encode(corpus)
    decoded = tok.decode(ids)
    assert "def foo" in decoded, f"decode failed: {repr(decoded)}"
    print(f"  [PASS] CharTokenizer  vocab={tok.vocab_size}  ids={len(ids)}")


def test_normalise_indentation():
    from tokenizer.bpe import normalise_indentation
    src = "def foo():\n    x = 1\n    return x\n"
    norm = normalise_indentation(src)
    assert "__INDENT__" in norm or "def foo" in norm  # either way, no crash
    print("  [PASS] normalise_indentation")


# ──────────────────────────────────────────────
# PyTorch model
# ──────────────────────────────────────────────

def test_torch_model_forward():
    from torch_impl.model import PythonCodeTransformer, TransformerConfig

    cfg = TransformerConfig(
        vocab_size=100, d_model=32, num_heads=4,
        num_layers=2, d_ff=64, max_seq_len=64
    )
    model  = PythonCodeTransformer(cfg)
    params = model.param_count()
    assert params > 0
    print(f"  [PASS] PythonCodeTransformer  params={params:,}")

    x      = torch.randint(0, 100, (2, 16))
    y      = torch.randint(0, 100, (2, 16))
    logits, loss = model(x, labels=y)
    assert logits.shape == (2, 16, 100)
    assert loss is not None and loss.item() > 0
    print(f"  [PASS] forward pass  logits={logits.shape}  loss={loss.item():.4f}")


def test_torch_model_generate():
    from torch_impl.model import PythonCodeTransformer, TransformerConfig

    cfg    = TransformerConfig(vocab_size=50, d_model=32, num_heads=4,
                                num_layers=2, d_ff=64)
    model  = PythonCodeTransformer(cfg)
    model.eval()
    x      = torch.tensor([[1, 2, 3]])
    logits = model.generate_logits(x)
    assert logits.shape == (1, 50)
    print(f"  [PASS] generate_logits  shape={logits.shape}")


def test_rope_embeddings():
    from torch_impl.model import RotaryEmbedding, apply_rope
    rope  = RotaryEmbedding(dim=16, max_seq_len=64)
    cos, sin = rope(8)
    assert cos.shape == (1, 1, 8, 16)
    q = torch.randn(1, 4, 8, 16)
    k = torch.randn(1, 4, 8, 16)
    q_r, k_r = apply_rope(q, k, cos, sin)
    assert q_r.shape == q.shape
    print("  [PASS] RoPE embeddings")


# ──────────────────────────────────────────────
# Data dataset
# ──────────────────────────────────────────────

def test_python_code_dataset():
    from tokenizer.bpe import CharTokenizer
    from data.dataset import PythonCodeDataset

    tok = CharTokenizer()
    tok.build_vocab("def foo():\n    return 42\n")

    codes   = ["def foo():\n    return 42\n", "x = 1\ny = 2\n"]
    windows = []
    ctx_len = 32
    for code in codes:
        ids    = tok.encode(code, add_bof=True, add_eof=True)
        padded = ids[:ctx_len + 1] + [0] * max(0, ctx_len + 1 - len(ids))
        windows.append(padded)

    ds = PythonCodeDataset(windows, context_len=ctx_len)
    x, y = ds[0]
    assert x.shape == (ctx_len,)
    assert y.shape == (ctx_len,)
    assert (x[1:] == y[:-1]).all()   # shift-by-one
    print(f"  [PASS] PythonCodeDataset  len={len(ds)}  x={x.shape}")


# ──────────────────────────────────────────────
# Inference (no checkpoint needed — uses random weights)
# ──────────────────────────────────────────────

def test_sample_token():
    from inference.suggest import sample_token
    logits   = torch.randn(100)
    tok, lp  = sample_token(logits, temperature=0.8, top_p=0.9)
    assert 0 <= tok < 100
    assert lp <= 0.0
    print(f"  [PASS] sample_token  tok={tok}  lp={lp:.3f}")


def test_is_valid_python():
    from inference.suggest import _is_valid_python
    assert     _is_valid_python("def foo():\n", "    return 42")
    assert not _is_valid_python("def foo():\n", "    return ===")
    print("  [PASS] _is_valid_python")


def test_edit_distance():
    from inference.suggest import _edit_distance
    assert _edit_distance("", "")   == 0
    assert _edit_distance("abc", "abc") == 0
    assert _edit_distance("abc", "abd") == 1
    assert _edit_distance("kitten", "sitting") == 3
    print("  [PASS] _edit_distance")


def test_deduplicate():
    from inference.suggest import _deduplicate, Suggestion
    sug = lambda c, s: Suggestion(completion=c, score=s, log_prob=-1.0,
                                   is_valid_py=True, latency_ms=10.0)
    suggestions = [
        sug("def foo():\n    pass", 0.9),
        sug("def foo():\n    pass", 0.8),   # exact dup
        sug("def bar():\n    return 1", 0.7),
    ]
    deduped = _deduplicate(suggestions)
    assert len(deduped) == 2
    print(f"  [PASS] _deduplicate  {len(suggestions)} → {len(deduped)}")


# ──────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────

ALL_TESTS = [
    test_causal_mask,
    test_core_forward,
    test_char_tokenizer,
    test_normalise_indentation,
    test_rope_embeddings,
    test_torch_model_forward,
    test_torch_model_generate,
    test_python_code_dataset,
    test_sample_token,
    test_is_valid_python,
    test_edit_distance,
    test_deduplicate,
]

if __name__ == "__main__":
    passed, failed = 0, 0
    print("\n=== Smoke Tests ===\n")
    for test_fn in ALL_TESTS:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {test_fn.__name__}: {e}")
            failed += 1
    print(f"\nResults: {passed} passed, {failed} failed")
    sys.exit(failed)
