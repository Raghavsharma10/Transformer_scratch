import torch
from torch_impl.transformer_encoder import TransformerEncoder

def test_transformer_encoder_shape():
    vocab_size = 50
    d = 16
    T = 7
    num_layers = 2

    token_ids = torch.randint(0, vocab_size, (T,))
    model = TransformerEncoder(vocab_size, d, num_layers)

    out = model(token_ids)
    assert out.shape == (T, d)