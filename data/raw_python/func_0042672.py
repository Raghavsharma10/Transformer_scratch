def moment(inlist, moment=1):
    """
Calculates the nth moment about the mean for a sample (defaults to
the 1st moment).  Used to calculate coefficients of skewness and kurtosis.

Usage:   lmoment(inlist,moment=1)
Returns: appropriate moment (r) from pyLibrary. 1/n * SUM((inlist(i)-mean)**r)
"""
    if moment == 1:
        return 0.0
    else:
        mn = mean(inlist)
        n = len(inlist)
        s = 0
        for x in inlist:
            s = s + (x - mn) ** moment
        return s / float(n)