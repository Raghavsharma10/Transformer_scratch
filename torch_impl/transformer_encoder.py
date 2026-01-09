import torch.nn as nn
from torch_impl.embeddings import InputEmbedding
from torch_impl.encoder import Encoder

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d, num_layers, max_len=5000):
        super().__init__()
        self.embedding = InputEmbedding(vocab_size, d, max_len)
        self.encoder = Encoder(d, num_layers)

    def forward(self, token_ids):
        """
        token_ids: (T,)
        returns: (T, d)
        """
        X = self.embedding(token_ids)  # (T, d)
        X = self.encoder(X)            # (T, d)
        return X