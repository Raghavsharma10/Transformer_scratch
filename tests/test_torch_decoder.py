import torch
from torch_impl.decoder import Decoder

def test_decoder_shape():
    d = 8
    T = 6
    num_layers = 3

    X = torch.randn(T, d)
    mask = torch.tril(torch.ones(T, T))

    decoder = Decoder(d, num_layers)
    out = decoder(X, mask)

    assert out.shape == (T, d)