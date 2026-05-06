def lincc(x, y):
    """
Calculates Lin's concordance correlation coefficient.

Usage:   alincc(x,y)    where x, y are equal-length arrays
Returns: Lin's CC
"""
    covar = cov(x, y) * (len(x) - 1) / float(len(x))  # correct denom to n
    xvar = var(x) * (len(x) - 1) / float(len(x))  # correct denom to n
    yvar = var(y) * (len(y) - 1) / float(len(y))  # correct denom to n
    lincc = (2 * covar) / ((xvar + yvar) + ((mean(x) - mean(y)) ** 2))
    return lincc