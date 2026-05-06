def list2doe(alist):
    """list2doe"""
    theequal = ''
    astr = ''
    lenj = len(alist)
    leni = len(alist[0])
    for i in range(0, leni-1):
        for j in range(0, lenj):
            if j == 0:
                astr = astr + alist[j][i + 1] + theequal + alist[j][0] + RET
            else:
                astr = astr + alist[j][0] + theequal + alist[j][i + 1] + RET
        astr = astr + RET
    return astr