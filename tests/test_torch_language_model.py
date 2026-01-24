import torch
from torch_impl.language_model import TransformerLM

def test_transformer_lm_shape():
    vocab_size = 50
    d = 16
    num_layers = 2
    T = 7

    token_ids = torch.randint(0, vocab_size, (T,))
    model = TransformerLM(vocab_size, d, num_layers)

    logits = model(token_ids)

    assert logits.shape == (T, vocab_size)