"""
Self-Attention (single head)

Input:
- X: (T, d) input sequence

Learned parameters:
- Wq, Wk, Wv ∈ R^(d x d)

Output:
- output: (T, d)

Logic:
- Q = X @ Wq
- K = X @ Wk
- V = X @ Wv
- output = scaled_dot_product_attention(Q, K, V)
"""

import random

from core.attention import scaled_dot_product_attention


class selfAttention:

    def __init__(self, d):
        
        self.d = d
        self.Wq = self._init_matrix(d,d)
        self.Wk = self._init_matrix(d,d)
        self.Wv = self._init_matrix(d,d)

    def _init_matrix(self, rows, cols):

        return [[random.uniform(-1,1) for _ in range (cols)] for _ in range(rows)]
    
    def forward(self, X):
        
        d = self.d 
        Wq = self.Wq
        Wk = self.Wk
        Wv = self.Wv
        T = len(X)

        Q = [[0.0 for _ in range(d)] for _ in range(T)]

        for i in range(T):
            for j in range(d):
                dot = 0.0
                for k in range(d):
                   dot += X[i][k]*Wq[k][j]
                Q[i][j] = dot       

        K = [[0.0 for _ in range(d)] for _ in range(T)] 

        for i in range(T):
            for j in range(d):
                dot = 0.0
                for k in range(d):
                   dot += X[i][k]*Wk[k][j]
                K[i][j] = dot

        V = [[0.0 for _ in range(d)] for _ in range(T)]


        for i in range(T):
            for j in range(d):
                dot = 0.0
                for k in range(d):
                   dot += X[i][k]*Wv[k][j]
                V[i][j] = dot

        
        output = scaled_dot_product_attention(Q,K,V)
        return output