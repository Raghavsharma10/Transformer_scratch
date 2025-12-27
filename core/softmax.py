"""
Softmax activation function implementation.


Input -> x: 1D list or 1D numpy array.

Output: 
probabilities having the same shape as x
sum of probabilites = 1

"""

import math


def softmax(x):
    """
This functions computes the softmax over a 1D array.
    """

    max_val = max(x)

    stabilized = [i - max_val for i in x]

    exponentiated = [math.exp(i) for i in stabilized]

    total = sum(exponentiated)

    probs = [i / total for i in exponentiated]

    assert abs(sum(probs) - 1.0) < 1e-6
    assert all(i>=0 for i in probs)

    return probs




