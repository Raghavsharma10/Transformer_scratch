"""
core/transformer.py
Original NumPy decoder-only transformer — preserved as reference implementation.
For production training, use torch_impl/model.py instead.
"""

import numpy as np


# ──────────────────────────────────────────────
# Utility
# ──────────────────────────────────────────────

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return gamma * (x - mean) / np.sqrt(var + eps) + beta


# ──────────────────────────────────────────────
# Positional encoding
# ──────────────────────────────────────────────

def sinusoidal_positional_encoding(seq_len: int, d_model: int) -> np.ndarray:
    pe = np.zeros((seq_len, d_model))
    pos = np.arange(seq_len)[:, None]
    div = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
    pe[:, 0::2] = np.sin(pos * div)
    pe[:, 1::2] = np.cos(pos * div)
    return pe  # (seq_len, d_model)


# ──────────────────────────────────────────────
# Attention
# ──────────────────────────────────────────────

def causal_mask(seq_len: int) -> np.ndarray:
    """Upper-triangular -inf mask so token i cannot attend to token j > i."""
    mask = np.triu(np.full((seq_len, seq_len), -1e9), k=1)
    return mask  # (seq_len, seq_len)


def multi_head_attention(
    x: np.ndarray,
    Wq: np.ndarray, Wk: np.ndarray, Wv: np.ndarray, Wo: np.ndarray,
    bq: np.ndarray, bk: np.ndarray, bv: np.ndarray, bo: np.ndarray,
    num_heads: int,
) -> np.ndarray:
    """
    x : (batch, seq, d_model)
    Returns : (batch, seq, d_model)
    """
    batch, seq, d_model = x.shape
    d_head = d_model // num_heads

    Q = x @ Wq + bq  # (B, S, D)
    K = x @ Wk + bk
    V = x @ Wv + bv

    # Split into heads → (B, H, S, d_head)
    def split_heads(t):
        t = t.reshape(batch, seq, num_heads, d_head)
        return t.transpose(0, 2, 1, 3)

    Q, K, V = split_heads(Q), split_heads(K), split_heads(V)

    scale = np.sqrt(d_head)
    scores = Q @ K.transpose(0, 1, 3, 2) / scale   # (B, H, S, S)
    scores += causal_mask(seq)                        # apply causal mask
    weights = softmax(scores, axis=-1)

    ctx = weights @ V                                 # (B, H, S, d_head)
    ctx = ctx.transpose(0, 2, 1, 3).reshape(batch, seq, d_model)
    return ctx @ Wo + bo


# ──────────────────────────────────────────────
# Feed-forward
# ──────────────────────────────────────────────

def feed_forward(x: np.ndarray, W1, b1, W2, b2) -> np.ndarray:
    return gelu(x @ W1 + b1) @ W2 + b2


# ──────────────────────────────────────────────
# Transformer parameters container
# ──────────────────────────────────────────────

class TransformerParams:
    """Holds all learnable parameters for the numpy transformer."""

    def __init__(self, vocab_size: int, d_model: int, num_heads: int,
                 num_layers: int, d_ff: int, max_seq: int = 512):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.max_seq = max_seq

        scale = 0.02
        # Token embedding
        self.token_emb = np.random.randn(vocab_size, d_model) * scale

        # Per-layer parameters
        self.layers = []
        for _ in range(num_layers):
            layer = {
                # Attention
                "Wq": np.random.randn(d_model, d_model) * scale,
                "Wk": np.random.randn(d_model, d_model) * scale,
                "Wv": np.random.randn(d_model, d_model) * scale,
                "Wo": np.random.randn(d_model, d_model) * scale,
                "bq": np.zeros(d_model),
                "bk": np.zeros(d_model),
                "bv": np.zeros(d_model),
                "bo": np.zeros(d_model),
                # FFN
                "W1": np.random.randn(d_model, d_ff) * scale,
                "b1": np.zeros(d_ff),
                "W2": np.random.randn(d_ff, d_model) * scale,
                "b2": np.zeros(d_model),
                # LayerNorm 1 & 2
                "ln1_g": np.ones(d_model), "ln1_b": np.zeros(d_model),
                "ln2_g": np.ones(d_model), "ln2_b": np.zeros(d_model),
            }
            self.layers.append(layer)

        # Final layer norm + LM head
        self.ln_f_g = np.ones(d_model)
        self.ln_f_b = np.zeros(d_model)
        self.lm_head = np.random.randn(d_model, vocab_size) * scale


# ──────────────────────────────────────────────
# Forward pass
# ──────────────────────────────────────────────

def forward(params: TransformerParams, token_ids: np.ndarray) -> np.ndarray:
    """
    token_ids : (batch, seq)  int32
    Returns   : (batch, seq, vocab_size)  logits
    """
    batch, seq = token_ids.shape
    x = params.token_emb[token_ids]                  # (B, S, D)
    x = x + sinusoidal_positional_encoding(seq, params.d_model)

    for layer in params.layers:
        # Pre-norm attention
        x_norm = layer_norm(x, layer["ln1_g"], layer["ln1_b"])
        attn_out = multi_head_attention(
            x_norm,
            layer["Wq"], layer["Wk"], layer["Wv"], layer["Wo"],
            layer["bq"], layer["bk"], layer["bv"], layer["bo"],
            params.num_heads,
        )
        x = x + attn_out

        # Pre-norm FFN
        x_norm = layer_norm(x, layer["ln2_g"], layer["ln2_b"])
        ffn_out = feed_forward(x_norm, layer["W1"], layer["b1"], layer["W2"], layer["b2"])
        x = x + ffn_out

    x = layer_norm(x, params.ln_f_g, params.ln_f_b)
    logits = x @ params.lm_head                      # (B, S, V)
    return logits


def cross_entropy_loss(logits: np.ndarray, targets: np.ndarray) -> float:
    """
    logits  : (batch, seq, vocab_size)
    targets : (batch, seq)
    """
    B, S, V = logits.shape
    logits_flat = logits.reshape(B * S, V)
    targets_flat = targets.reshape(B * S)
    log_probs = logits_flat - np.log(np.exp(logits_flat).sum(axis=-1, keepdims=True) + 1e-9)
    loss = -log_probs[np.arange(B * S), targets_flat].mean()
    return float(loss)
