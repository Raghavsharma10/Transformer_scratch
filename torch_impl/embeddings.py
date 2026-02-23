import torch
import torch.nn as nn
import math


class TokenEmbedding(nn.Module):

    def __init__(self, vocab_size, d):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d)
        self.d = d

    def forward(self, token_ids):
        """
        token_ids: (T,)
        returns: (T, d)
        """
        return self.embedding(token_ids) * math.sqrt(self.d) 
    
    # Note : self.embedding(token_ids) is a short hand of self.embedding.forward(token_ids)
    # This is a pytorch rule - layer(x) --> layer.forward(x)

class PositionalEncoding(nn.Module):
    def __init__(self, d, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d)
        position = torch.arange(0, max_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d, 2) * (-math.log(10000.0) / d)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, X):
        """
        X: (T, d)
        returns: (T, d)
        """
        T = X.size(0)
        return X + self.pe[:T]
    
class InputEmbedding(nn.Module):
    def __init__(self, vocab_size, d, max_len=5000):
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, d)
        self.positional_encoding = PositionalEncoding(d, max_len)

    def forward(self, token_ids):
        """
        token_ids: (T,)
        returns: (T, d)
        """
        X = self.token_embedding(token_ids)
        X = self.positional_encoding(X)
        return X