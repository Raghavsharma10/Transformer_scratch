def harmonicmean(inlist):
    """
Calculates the harmonic mean of the values in the passed list.
That is:  n / (1/x1 + 1/x2 + ... + 1/xn).  Assumes a '1D' list.

Usage:   lharmonicmean(inlist)
"""
    sum = 0
    for item in inlist:
        sum = sum + 1.0 / item
    return len(inlist) / sum