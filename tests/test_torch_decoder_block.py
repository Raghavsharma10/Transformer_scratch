import torch
from torch_impl.decoder_block import DecoderBlock

def test_decoder_block_shape():
    d = 8
    T = 5

    X = torch.randn(T, d)

    # causal mask (lower triangular)
    mask = torch.tril(torch.ones(T, T))

    block = DecoderBlock(d)
    out = block(X, mask)

    assert out.shape == (T, d)