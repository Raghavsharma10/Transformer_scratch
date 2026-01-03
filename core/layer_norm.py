class LayerNorm:
    def __init__(self, d, eps=1e-5):
        self.d = d
        self.eps = eps
        self.gamma = [1.0] * d
        self.beta = [0.0] * d

    def forward(self, X):
        out = []
        for x in X:
            mean = sum(x) / self.d
            var = sum((xi - mean) ** 2 for xi in x) / self.d
            std = (var + self.eps) ** 0.5

            y = [
                self.gamma[i] * ((x[i] - mean) / std) + self.beta[i]
                for i in range(self.d)
            ]
            out.append(y)
        return out