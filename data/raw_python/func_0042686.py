def trimboth(l, proportiontocut):
    """
Slices off the passed proportion of items from BOTH ends of the passed
list (i.e., with proportiontocut=0.1, slices 'leftmost' 10% AND 'rightmost'
10% of scores.  Assumes list is sorted by magnitude.  Slices off LESS if
proportion results in a non-integer slice index (i.e., conservatively
slices off proportiontocut).

Usage:   ltrimboth (l,proportiontocut)
Returns: trimmed version of list l
"""
    lowercut = int(proportiontocut * len(l))
    uppercut = len(l) - lowercut
    return l[lowercut:uppercut]