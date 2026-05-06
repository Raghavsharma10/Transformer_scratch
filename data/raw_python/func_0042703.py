def sumdiffsquared(x, y):
    """
Takes pairwise differences of the values in lists x and y, squares
these differences, and returns the sum of these squares.

Usage:   lsumdiffsquared(x,y)
Returns: sum[(x[i]-y[i])**2]
"""
    sds = 0
    for i in range(len(x)):
        sds = sds + (x[i] - y[i]) ** 2
    return sds