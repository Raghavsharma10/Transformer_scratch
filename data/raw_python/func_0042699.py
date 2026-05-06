def incr(l, cap):        # to increment a list up to a max-list of 'cap'
    """
Simulate a counting system from an n-dimensional list.

Usage:   lincr(l,cap)   l=list to increment, cap=max values for each list pos'n
Returns: next set of values for list l, OR -1 (if overflow)
"""
    l[0] = l[0] + 1     # e.g., [0,0,0] --> [2,4,3] (=cap)
    for i in range(len(l)):
        if l[i] > cap[i] and i < len(l) - 1: # if carryover AND not done
            l[i] = 0
            l[i + 1] = l[i + 1] + 1
        elif l[i] > cap[i] and i == len(l) - 1: # overflow past last column, must be finished
            l = -1
    return l