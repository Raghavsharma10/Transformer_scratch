def sem(inlist):
    """
Returns the estimated standard error of the mean (sx-bar) of the
values in the passed list.  sem = stdev / sqrt(n)

Usage:   lsem(inlist)
"""
    sd = stdev(inlist)
    n = len(inlist)
    return sd / math.sqrt(n)