def medianscore(inlist):
    """
Returns the 'middle' score of the passed list.  If there is an even
number of scores, the mean of the 2 middle scores is returned.

Usage:   lmedianscore(inlist)
"""

    newlist = copy.deepcopy(inlist)
    newlist.sort()
    if len(newlist) % 2 == 0:   # if even number of scores, average middle 2
        index = len(newlist) / 2  # integer division correct
        median = float(newlist[index] + newlist[index - 1]) / 2
    else:
        index = len(newlist) / 2  # int divsion gives mid value when count from 0
        median = newlist[index]
    return median