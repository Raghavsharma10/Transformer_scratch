def var(inlist):
    """
Returns the variance of the values in the passed list using N-1
for the denominator (i.e., for estimating population variance).

Usage:   lvar(inlist)
"""
    n = len(inlist)
    mn = mean(inlist)
    deviations = [0] * len(inlist)
    for i in range(len(inlist)):
        deviations[i] = inlist[i] - mn
    return ss(deviations) / float(n - 1)