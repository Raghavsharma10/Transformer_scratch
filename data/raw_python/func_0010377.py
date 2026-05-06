def morphAveragePitch(fromDataList, toDataList):
    '''
    Adjusts the values in fromPitchList to have the same average as toPitchList
    
    Because other manipulations can alter the average pitch, morphing the pitch
    is the last pitch manipulation that should be done
    
    After the morphing, the code removes any values below zero, thus the
    final average might not match the target average.
    '''
    
    timeList, fromPitchList = zip(*fromDataList)
    toPitchList = [pitchVal for _, pitchVal in toDataList]
    
    # Zero pitch values aren't meaningful, so filter them out if they are
    # in the dataset
    fromListNoZeroes = [val for val in fromPitchList if val > 0]
    fromAverage = sum(fromListNoZeroes) / float(len(fromListNoZeroes))
    
    toListNoZeroes = [val for val in toPitchList if val > 0]
    toAverage = sum(toListNoZeroes) / float(len(toListNoZeroes))
    
    newPitchList = [val - fromAverage + toAverage for val in fromPitchList]
    
#     finalAverage = sum(newPitchList) / float(len(newPitchList))
    
    # Removing zeroes and negative pitch values
    retDataList = [(time, pitchVal) for time, pitchVal
                   in zip(timeList, newPitchList)
                   if pitchVal > 0]
    
    return retDataList