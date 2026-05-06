def getManipulatedParamaters(tgFN, tierName, modFunc,
                             filterFunc=None, useBlanks=False):
    '''
    Get intervals for source and target audio files
    
    Use this information to find out how much to stretch/shrink each source
    interval.
    
    The target values are based on modfunc.
    '''
    
    fromExtractInfo = utils.getIntervals(tgFN, tierName, filterFunc,
                                         useBlanks)
    
    durationParameters = []
    for fromInfoTuple in fromExtractInfo:
        fromStart, fromEnd = fromInfoTuple[:2]
        toStart, toEnd = modFunc(fromStart), modFunc(fromEnd)

        # Praat will ignore a second value appearing at the same time as
        # another so we give each start a tiny offset to distinguish intervals
        # that start and end at the same point
        toStart += PRAAT_TIME_DIFF
        fromStart += PRAAT_TIME_DIFF

        ratio = (toEnd - toStart) / float((fromEnd - fromStart))

        ratioTuple = (fromStart, fromEnd, ratio)
        durationParameters.append(ratioTuple)

    return durationParameters