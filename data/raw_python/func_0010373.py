def _getSmallestDifference(inputList, targetVal):
    '''
    Returns the value in inputList that is closest to targetVal
    
    Iteratively splits the dataset in two, so it should be pretty fast
    '''
    targetList = inputList[:]
    retVal = None
    while True:
        # If we're down to one value, stop iterating
        if len(targetList) == 1:
            retVal = targetList[0]
            break
        halfPoint = int(len(targetList) / 2.0) - 1
        a = targetList[halfPoint]
        b = targetList[halfPoint + 1]
        
        leftDiff = abs(targetVal - a)
        rightDiff = abs(targetVal - b)
        
        # If the distance is 0, stop iterating, the targetVal is present
        # in the inputList
        if leftDiff == 0 or rightDiff == 0:
            retVal = targetVal
            break
        
        # Look at left half or right half
        if leftDiff < rightDiff:
            targetList = targetList[:halfPoint + 1]
        else:
            targetList = targetList[halfPoint + 1:]
         
    return retVal