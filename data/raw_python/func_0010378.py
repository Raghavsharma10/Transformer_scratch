def morphRange(fromDataList, toDataList):
    '''
    Changes the scale of values in one distribution to that of another
    
    ie The maximum value in fromDataList will be set to the maximum value in
    toDataList.  The 75% largest value in fromDataList will be set to the
    75% largest value in toDataList, etc.
    
    Small sample sizes will yield results that are not very meaningful
    '''
    
    # Isolate and sort pitch values
    fromPitchList = [dataTuple[1] for dataTuple in fromDataList]
    toPitchList = [dataTuple[1] for dataTuple in toDataList]
    
    fromPitchListSorted = sorted(fromPitchList)
    toPitchListSorted = sorted(toPitchList)
    
    # Bin pitch values between 0 and 1
    fromListRel = makeSequenceRelative(fromPitchListSorted)[0]
    toListRel = makeSequenceRelative(toPitchListSorted)[0]
    
    # Find each values closest equivalent in the other list
    indexList = _getNearestMappingIndexList(fromListRel, toListRel)
    
    # Map the source pitch to the target pitch value
    # Pitch value -> get sorted position -> get corresponding position in
    # target list -> get corresponding pitch value = the new pitch value
    retList = []
    for time, pitch in fromDataList:
        fromI = fromPitchListSorted.index(pitch)
        toI = indexList[fromI]
        newPitch = toPitchListSorted[toI]
        
        retList.append((time, newPitch))
    
    return retList