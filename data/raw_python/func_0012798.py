def getoneblock(astr, start, end):
    """get the block bounded by start and end
    doesn't work for multiple blocks"""
    alist = astr.split(start)
    astr = alist[-1]
    alist = astr.split(end)
    astr = alist[0]
    return astr