def slicedIterator(sourceList, sliceSize):
    """
    :param: sourceList: list which need to be sliced
    :type: list
    :param: sliceSize: size of the slice
    :type: int
    :return: iterator of the sliced list
    """
    start = 0
    end = 0

    while len(sourceList) > end:
        end = start + sliceSize
        yield sourceList[start: end]
        start = end