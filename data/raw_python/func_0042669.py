def mean(inlist):
    """
Returns the arithematic mean of the values in the passed list.
Assumes a '1D' list, but will function on the 1st dim of an array(!).

Usage:   lmean(inlist)
"""
    sum = 0
    for item in inlist:
        sum = sum + item
    return sum / float(len(inlist))