import torch
from torch_impl.output_head import OutputHead


def test_output_head_shape():
    d = 16
    vocab_size = 50
    T = 7

    X = torch.randn(T, d)
    head = OutputHead(d, vocab_size)

    logits = head.forward(X)

    assert logits.shape == (T, vocab_size)

    
