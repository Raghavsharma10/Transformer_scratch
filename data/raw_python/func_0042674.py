def itemfreq(inlist):
    """
Returns a list of pairs.  Each pair consists of one of the scores in inlist
and it's frequency count.  Assumes a 1D list is passed.

Usage:   litemfreq(inlist)
Returns: a 2D frequency table (col [0:n-1]=scores, col n=frequencies)
"""
    scores = pstat.unique(inlist)
    scores.sort()
    freq = []
    for item in scores:
        freq.append(inlist.count(item))
    return zip(scores, freq)