def describe(inlist):
    """
Returns some descriptive statistics of the passed list (assumed to be 1D).

Usage:   ldescribe(inlist)
Returns: n, mean, standard deviation, skew, kurtosis
"""
    n = len(inlist)
    mm = (min(inlist), max(inlist))
    m = mean(inlist)
    sd = stdev(inlist)
    sk = skew(inlist)
    kurt = kurtosis(inlist)
    return n, mm, m, sd, sk, kurt