def geometricmean(inlist):
    """
Calculates the geometric mean of the values in the passed list.
That is:  n-th root of (x1 * x2 * ... * xn).  Assumes a '1D' list.

Usage:   lgeometricmean(inlist)
"""
    mult = 1.0
    one_over_n = 1.0 / len(inlist)
    for item in inlist:
        mult = mult * pow(item, one_over_n)
    return mult