def morphChunkedDataLists(fromDataList, toDataList, stepList):
    '''
    Morph one set of data into another, in a stepwise fashion

    A convenience function.  Given a set of paired data lists,
    this will morph each one individually.

    Returns a single list with all data combined together.
    '''

    assert(len(fromDataList) == len(toDataList))

    # Morph the fromDataList into the toDataList
    outputList = []
    for x, y in zip(fromDataList, toDataList):

        # We cannot morph a region if there is no data or only
        # a single data point for either side
        if (len(x) < 2) or (len(y) < 2):
            continue

        tmpList = [outputPitchList for _, outputPitchList
                   in morphDataLists(x, y, stepList)]
        outputList.append(tmpList)

    # Transpose list
    finalOutputList = outputList.pop(0)
    for subList in outputList:
        for i, subsubList in enumerate(subList):
            finalOutputList[i].extend(subsubList)

    return finalOutputList