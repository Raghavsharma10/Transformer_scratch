import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):

    def __init__(self, d, num_heads):
        super().__init__()

        assert d % num_heads == 0

        self.num_heads = num_heads
        self.d_head = d // num_heads

        self.Wq = nn.Linear(d, d)
        self.Wk = nn.Linear(d, d)
        self.Wv = nn.Linear(d, d)

        self.Wo = nn.Linear(d, d)

    def forward(self, X, mask=None):

        T = X.size(0)

        Q = self.Wq(X)
        K = self.Wk(X)
        V = self.Wv(X)

        # split into heads
        Q = Q.view(T, self.num_heads, self.d_head).transpose(0,1)
        K = K.view(T, self.num_heads, self.d_head).transpose(0,1)
        V = V.view(T, self.num_heads, self.d_head).transpose(0,1)

        scores = Q @ K.transpose(-2, -1)
        scores /= math.sqrt(self.d_head)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        out = attn @ V

        # concat heads
        out = out.transpose(0,1).contiguous().view(T, -1)

        return self.Wo(out)