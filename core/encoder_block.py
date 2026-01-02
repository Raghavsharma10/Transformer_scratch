from core.self_attention import selfAttention

def layer_norm(X):
    """
    Placeholder for Layer Normalization.
    For now, return x unchanged.
    """
    return X

def feed_forward(X):
    """
    Placeholder for Feed-Forward Network.
    For now, return x unchanged.
    """
    return X

class Encoderblock:

    def __init__(self, d):
        self.self_attention = selfAttention(d)

    def forward(self, X):
        attn_out = self.self_attention.forward(X)

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
    

        X = layer_norm(X)

        return X
    