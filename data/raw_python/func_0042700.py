def cumsum(inlist):
    """
Returns a list consisting of the cumulative sum of the items in the
passed list.

Usage:   lcumsum(inlist)
"""
    newlist = copy.deepcopy(inlist)
    for i in range(1, len(newlist)):
        newlist[i] = newlist[i] + newlist[i - 1]
    return newlist