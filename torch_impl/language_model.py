import torch.nn as nn
from torch_impl.embeddings import InputEmbedding
from torch_impl.encoder import Encoder
from torch_impl.output_head import OutputHead

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d, num_layers, max_len=5000):
        super().__init__()
        self.embed = InputEmbedding(vocab_size, d, max_len)
        self.encoder = Encoder(d, num_layers)
        self.head = OutputHead(d, vocab_size)

    def forward(self, token_ids):
        """
        token_ids: (T,)
        returns: logits (T, vocab_size)
        """
        X = self.embed(token_ids)      # (T, d)
        X = self.encoder(X)            # (T, d)
        logits = self.head(X)          # (T, vocab_size)
        return logits