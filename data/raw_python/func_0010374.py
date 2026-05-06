def _getNearestMappingIndexList(fromValList, toValList):
    '''
    Finds the indicies for data points that are closest to each other.

    The inputs should be in relative time, scaled from 0 to 1
    e.g. if you have [0, .1, .5., .9] and [0, .1, .2, 1]
    will output [0, 1, 1, 2]
    '''

    indexList = []
    for fromTimestamp in fromValList:
        smallestDiff = _getSmallestDifference(toValList, fromTimestamp)
        i = toValList.index(smallestDiff)
        indexList.append(i)

    return indexList