from core.self_attention import selfAttention
from core.attention import causal_mask
from core.layer_norm import LayerNorm
from core.feed_forward import FeedForward

class Decoderblock:

    def __init__(self, d):
        self.d = d
        self.self_attention = selfAttention(d)
        self.norm1 = LayerNorm(d)
        self.norm2 = LayerNorm(d)
        self.ff = FeedForward(d)

    def forward(self, X):
        T = len(X)

        mask = causal_mask(T)

        attn_out = self.self_attention.forward(X, mask = mask)
    
        X = [
            [X[i][j] + attn_out[i][j] for j in range(len(X[0]))]
            for i in range(len(X))
        ]

        X = self.norm1.forward(X)

        ff_out = self.ff.forward(X)

        X = [
            [X[i][j] + ff_out[i][j] for j in range(len(X[0]))]
            for i in range(len(X))
            ]
        
        X = self.norm2.forward(X)

        return X
        



        