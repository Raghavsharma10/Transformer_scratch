import torch 
import torch.nn as nn
from torch_impl.self_attention import SelfAttention


class EncoderBlock(nn.Module):

    def __init__(self,d):
        super().__init__()

        self.self_attention = SelfAttention(d)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(
            nn.Linear(d, 4*d),
            nn.ReLU(),
            nn.Linear(4*d, d)
        )

    def forward(self, X):

        attn_out = self.self_attention(X)
        X = X + attn_out
        X = self.norm1(X)

        ff_out = self.ff(X)
        
        X = X + ff_out 
        X = self.norm2(X)

        return X

        