def percentileofscore(inlist, score, histbins=10, defaultlimits=None):
    """
Returns the percentile value of a score relative to the distribution
given by inlist.  Formula depends on the values used to histogram the data(!).

Usage:   lpercentileofscore(inlist,score,histbins=10,defaultlimits=None)
"""

    h, lrl, binsize, extras = histogram(inlist, histbins, defaultlimits)
    cumhist = cumsum(copy.deepcopy(h))
    i = int((score - lrl) / float(binsize))
    pct = (cumhist[i - 1] + ((score - (lrl + binsize * i)) / float(binsize)) * h[i]) / float(len(inlist)) * 100
    return pct