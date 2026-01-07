import torch 
import torch.nn as nn
import math

class SelfAttention(nn.Module):

    def __init__(self, d):
        super().__init__()
        self.Wq = nn.Linear(d,d)    #nn.Linear = parameter + matrix multiplication + bias
        self.Wk = nn.Linear(d,d)
        self.Wv = nn.Linear(d,d)
        self.d = d

    def forward(self, X, mask = None):

        Q = self.Wq(X)
        K = self.Wk(X)
        V = self.Wv(X)

        scores = Q @ K.transpose(-2, -1)
        scores /= math.sqrt(self.d)

        if mask is not None:
            scores = scores.masked_fill(mask ==0, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        out = attn @ V
        
        return out

