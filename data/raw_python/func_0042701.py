def ss(inlist):
    """
Squares each value in the passed list, adds up these squares and
returns the result.

Usage:   lss(inlist)
"""
    ss = 0
    for item in inlist:
        ss = ss + item * item
    return ss