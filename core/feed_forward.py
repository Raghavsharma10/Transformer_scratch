import random

class FeedForward:
    def __init__(self, d, hidden_mult=4):
        self.d = d
        self.h = hidden_mult * d

        self.W1 = [[random.uniform(-0.1, 0.1) for _ in range(self.h)] for _ in range(d)]
        self.b1 = [0.0] * self.h

        self.W2 = [[random.uniform(-0.1, 0.1) for _ in range(d)] for _ in range(self.h)]
        self.b2 = [0.0] * d

    def forward(self, X):
        out = []
        for x in X:
            # xW1 + b1
            h = []
            for j in range(self.h):
                s = self.b1[j]
                for i in range(self.d):
                    s += x[i] * self.W1[i][j]
                h.append(max(0.0, s))  # ReLU

            # hW2 + b2
            y = []
            for j in range(self.d):
                s = self.b2[j]
                for i in range(self.h):
                    s += h[i] * self.W2[i][j]
                y.append(s)

            out.append(y)
        return out