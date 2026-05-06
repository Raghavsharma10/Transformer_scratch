def zs(inlist):
    """
Returns a list of z-scores, one for each score in the passed list.

Usage:   lzs(inlist)
"""
    zscores = []
    for item in inlist:
        zscores.append(z(inlist, item))
    return zscores