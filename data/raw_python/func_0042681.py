def samplevar(inlist):
    """
Returns the variance of the values in the passed list using
N for the denominator (i.e., DESCRIBES the sample variance only).

Usage:   lsamplevar(inlist)
"""
    n = len(inlist)
    mn = mean(inlist)
    deviations = []
    for item in inlist:
        deviations.append(item - mn)
    return ss(deviations) / float(n)