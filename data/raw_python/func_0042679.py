def relfreq(inlist, numbins=10, defaultreallimits=None):
    """
Returns a relative frequency histogram, using the histogram function.

Usage:   lrelfreq(inlist,numbins=10,defaultreallimits=None)
Returns: list of cumfreq bin values, lowerreallimit, binsize, extrapoints
"""
    h, l, b, e = histogram(inlist, numbins, defaultreallimits)
    for i in range(len(h)):
        h[i] = h[i] / float(len(inlist))
    return h, l, b, e