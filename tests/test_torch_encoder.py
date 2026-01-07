import torch
from torch_impl.encoder import Encoder

def test_encoder_shape():
    d = 8
    T = 6
    num_layers = 4

    X = torch.randn(T, d)
    encoder = Encoder(d, num_layers)
    out = encoder(X)

    assert out.shape == (T, d)