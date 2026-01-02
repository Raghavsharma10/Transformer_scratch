from core.self_attention import selfAttention
from core.attention import causal_mask

def layer_norm(x):
        return x
    
def feed_forward(x):
        return x

class Decoderblock:

    def __init__(self, d):
        self.d = d
        self.self_attention = selfAttention(d)

    def forward(self, X):
        T = len(X)

        mask = causal_mask(T)

        attn_out = self.self_attention.forward(X, mask = mask)
    
        X = [
            [X[i][j] + attn_out[i][j] for j in range(len(X[0]))]
            for i in range(len(X))
        ]

        X = layer_norm(X)

        ff_out = feed_forward(X)

        X = [
            [X[i][j] + ff_out[i][j] for j in range(len(X[0]))]
            for i in range(len(X))
            ]

        return X
        



        