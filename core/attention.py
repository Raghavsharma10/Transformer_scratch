"""
Scaled Dot-Product Attention (single head)

Input: 

Q: (Tq, d)
K: (Tk, d)
V: (Tk, d)

Output:

output: (Tq, d)

Logic :

scores matrix = Q.K(T) / root(d)

Attention matrix(weights) = softmax(scores) row wise

Output = weights . V


Properties:
- attention weights to sum 1 across keys
- output is weighted sum of V
"""

import math
from core.softmax import softmax

def scaled_dot_product_attention(Q,K,V):

    Tq = len(Q)
    d = len(Q[0])
    Tk = len(K)

    """compute raw scores Q.Kt"""

    #initialising the score matrix

    scores = [[0.0 for _ in range(Tk)] for _ in range(Tq)]

    for i in range(Tq):         # matrix multiplication
        for j in range(Tk):
            dot = 0.0
            for k in range(d):
                dot += Q[i][k] * K[j][k]
            scores[i][j] = dot

    scale = math.sqrt(d)         #Dividing by the normalizing factor rootd

    for i in range(Tq):
        for j in range(Tk):
            scores[i][j] /= scale

    
    # applying softmax


    attention_weights = []          #Applying softmax row-wise

    for i in range(Tq):
        row = scores[i]
        probs = softmax(row)
        attention_weights.append(probs)
    
    # Multiplying the attention weights with V

    output = [[0.0 for _ in range(d)] for _ in range(Tq)]

    for i in range(Tq):
        for j in range(Tk):
            weight = attention_weights[i][j]
            for k in range(d):
                output[i][k] += weight * V[j][k]
    

        # invariants
    for row in attention_weights:
        assert abs(sum(row) - 1.0) < 1e-6
        
    return output


