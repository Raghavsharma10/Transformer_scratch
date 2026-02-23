import torch
from torch_impl.embeddings import InputEmbedding


def test_input_embedding_shape():
    vocab_size = 100
    d = 16
    T = 7

    token_ids = torch.randint(0, vocab_size, (T,))
    embed = InputEmbedding(vocab_size, d)

    out = embed(token_ids)

    assert out.shape == (T, d)