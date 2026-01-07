import torch
from torch_impl.encoder_block import EncoderBlock

def test_encoder_block_shape():
    d = 8
    T = 5
    X = torch.randn(T, d)

    block = EncoderBlock(d)
    out = block(X)

    assert out.shape == (T, d)

def test_encoder_block_has_parameters():
    block = EncoderBlock(8)
    params = list(block.parameters())
    assert len(params) > 0