def median(lst):
    """ Calcuates the median value in a @lst """
    #: http://stackoverflow.com/a/24101534
    sortedLst = sorted(lst)
    lstLen = len(lst)
    index = (lstLen - 1) // 2
    if (lstLen % 2):
        return sortedLst[index]
    else:
        return (sortedLst[index] + sortedLst[index + 1])/2.0