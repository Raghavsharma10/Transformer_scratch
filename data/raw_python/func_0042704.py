def shellsort(inlist):
    """
Shellsort algorithm.  Sorts a 1D-list.

Usage:   lshellsort(inlist)
Returns: sorted-inlist, sorting-index-vector (for original list)
"""
    n = len(inlist)
    svec = copy.deepcopy(inlist)
    ivec = range(n)
    gap = n / 2   # integer division needed
    while gap > 0:
        for i in range(gap, n):
            for j in range(i - gap, -1, -gap):
                while j >= 0 and svec[j] > svec[j + gap]:
                    temp = svec[j]
                    svec[j] = svec[j + gap]
                    svec[j + gap] = temp
                    itemp = ivec[j]
                    ivec[j] = ivec[j + gap]
                    ivec[j + gap] = itemp
        gap = gap / 2  # integer division needed
    # svec is now sorted inlist, and ivec has the order svec[i] = vec[ivec[i]]
    return svec, ivec