def trim1(l, proportiontocut, tail='right'):
    """
Slices off the passed proportion of items from ONE end of the passed
list (i.e., if proportiontocut=0.1, slices off 'leftmost' or 'rightmost'
10% of scores).  Slices off LESS if proportion results in a non-integer
slice index (i.e., conservatively slices off proportiontocut).

Usage:   ltrim1 (l,proportiontocut,tail='right')  or set tail='left'
Returns: trimmed version of list l
"""
    if tail == 'right':
        lowercut = 0
        uppercut = len(l) - int(proportiontocut * len(l))
    elif tail == 'left':
        lowercut = int(proportiontocut * len(l))
        uppercut = len(l)
    return l[lowercut:uppercut]