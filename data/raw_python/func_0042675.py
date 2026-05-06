def scoreatpercentile(inlist, percent):
    """
Returns the score at a given percentile relative to the distribution
given by inlist.

Usage:   lscoreatpercentile(inlist,percent)
"""
    if percent > 1:
        print("\nDividing percent>1 by 100 in lscoreatpercentile().\n")
        percent = percent / 100.0
    targetcf = percent * len(inlist)
    h, lrl, binsize, extras = histogram(inlist)
    cumhist = cumsum(copy.deepcopy(h))
    for i in range(len(cumhist)):
        if cumhist[i] >= targetcf:
            break
    score = binsize * ((targetcf - cumhist[i - 1]) / float(h[i])) + (lrl + binsize * i)
    return score