def morphDataLists(fromList, toList, stepList):
    '''
    Iteratively morph fromList into toList using the values 0 to 1 in stepList
    
    stepList: a value of 0 means no change and a value of 1 means a complete
    change to the other value
    '''

    # If there are more than 1 pitch value, then we align the data in
    # relative time.
    # Each data point comes with a timestamp.  The earliest timestamp is 0
    # and the latest timestamp is 1.  Using this method, for each relative
    # timestamp in the source list, we find the closest relative timestamp
    # in the target list.  Just because two pitch values have the same index
    # in the source and target lists does not mean that they correspond to
    # the same speech event.
    fromListRel, fromStartTime, fromEndTime = _makeTimingRelative(fromList)
    toListRel = _makeTimingRelative(toList)[0]

    # If fromList has more points, we'll have flat areas
    # If toList has more points, we'll might miss peaks or valleys
    fromTimeList = [dataTuple[0] for dataTuple in fromListRel]
    toTimeList = [dataTuple[0] for dataTuple in toListRel]
    indexList = _getNearestMappingIndexList(fromTimeList, toTimeList)
    alignedToPitchRel = [toListRel[i] for i in indexList]

    for stepAmount in stepList:
        newPitchList = []

        # Perform the interpolation
        for fromTuple, toTuple in zip(fromListRel, alignedToPitchRel):
            fromTime, fromValue = fromTuple
            toTime, toValue = toTuple

            # i + 1 b/c i_0 = 0 = no change
            newValue = fromValue + (stepAmount * (toValue - fromValue))
            newTime = fromTime + (stepAmount * (toTime - fromTime))

            newPitchList.append((newTime, newValue))

        newPitchList = _makeTimingAbsolute(newPitchList, fromStartTime,
                                           fromEndTime)

        yield stepAmount, newPitchList