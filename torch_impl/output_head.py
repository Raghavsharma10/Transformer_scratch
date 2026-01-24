import torch.nn as nn

class OutputHead(nn.Module):

    def __init__(self, d, vocab_size):
        super().__init__()
        self.d = d
        self.vocab_size = vocab_size
        self.proj = nn.Linear(d, vocab_size)

    def forward(self, X):

        return self.proj(X)

