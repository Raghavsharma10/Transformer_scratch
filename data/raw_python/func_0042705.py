def rankdata(inlist):
    """
Ranks the data in inlist, dealing with ties appropritely.  Assumes
a 1D inlist.  Adapted from Gary Perlman's |Stat ranksort.

Usage:   rankdata(inlist)
Returns: a list of length equal to inlist, containing rank scores
"""
    n = len(inlist)
    svec, ivec = shellsort(inlist)
    sumranks = 0
    dupcount = 0
    newlist = [0] * n
    for i in range(n):
        sumranks = sumranks + i
        dupcount = dupcount + 1
        if i == n - 1 or svec[i] != svec[i + 1]:
            averank = sumranks / float(dupcount) + 1
            for j in range(i - dupcount + 1, i + 1):
                newlist[ivec[j]] = averank
            sumranks = 0
            dupcount = 0
    return newlist