def getMorphParameters(fromTGFN, toTGFN, tierName,
                       filterFunc=None, useBlanks=False):
    '''
    Get intervals for source and target audio files
    
    Use this information to find out how much to stretch/shrink each source
    interval.
    
    The target values are based on the contents of toTGFN.
    '''
    
    if filterFunc is None:
        filterFunc = lambda entry: True  # Everything is accepted
    
    fromEntryList = utils.getIntervals(fromTGFN, tierName,
                                       includeUnlabeledRegions=useBlanks)
    toEntryList = utils.getIntervals(toTGFN, tierName,
                                     includeUnlabeledRegions=useBlanks)

    fromEntryList = [entry for entry in fromEntryList if filterFunc(entry)]
    toEntryList = [entry for entry in toEntryList if filterFunc(entry)]

    assert(len(fromEntryList) == len(toEntryList))

    durationParameters = []
    for fromEntry, toEntry in zip(fromEntryList, toEntryList):
        fromStart, fromEnd = fromEntry[:2]
        toStart, toEnd = toEntry[:2]

        # Praat will ignore a second value appearing at the same time as
        # another so we give each start a tiny offset to distinguish intervals
        # that start and end at the same point
        toStart += PRAAT_TIME_DIFF
        fromStart += PRAAT_TIME_DIFF
        
        ratio = (toEnd - toStart) / float((fromEnd - fromStart))
        durationParameters.append((fromStart, fromEnd, ratio))
    
    return durationParameters