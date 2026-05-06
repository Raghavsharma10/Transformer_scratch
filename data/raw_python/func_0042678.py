def cumfreq(inlist, numbins=10, defaultreallimits=None):
    """
Returns a cumulative frequency histogram, using the histogram function.

Usage:   lcumfreq(inlist,numbins=10,defaultreallimits=None)
Returns: list of cumfreq bin values, lowerreallimit, binsize, extrapoints
"""
    h, l, b, e = histogram(inlist, numbins, defaultreallimits)
    cumhist = cumsum(copy.deepcopy(h))
    return cumhist, l, b, e