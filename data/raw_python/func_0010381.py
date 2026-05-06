def changeDuration(fromWavFN, durationParameters, stepList, outputName,
                   outputMinPitch, outputMaxPitch, praatEXE):
    '''
    Uses praat to morph duration in one file to duration in another

    Praat uses the PSOLA algorithm
    '''

    rootPath = os.path.split(fromWavFN)[0]

    # Prep output directories
    outputPath = join(rootPath, "duration_resynthesized_wavs")
    utils.makeDir(outputPath)
    
    durationTierPath = join(rootPath, "duration_tiers")
    utils.makeDir(durationTierPath)

    fromWavDuration = audio_scripts.getSoundFileDuration(fromWavFN)

    durationParameters = copy.deepcopy(durationParameters)
    # Pad any gaps with values of 1 (no change in duration)
    
    # No need to stretch out any pauses at the beginning
    if durationParameters[0][0] != 0:
        tmpVar = (0, durationParameters[0][0] - PRAAT_TIME_DIFF, 1)
        durationParameters.insert(0, tmpVar)

    # Or the end
    if durationParameters[-1][1] < fromWavDuration:
        durationParameters.append((durationParameters[-1][1] + PRAAT_TIME_DIFF,
                                   fromWavDuration, 1))

    # Create the praat script for doing duration manipulation
    for stepAmount in stepList:
        durationPointList = []
        for start, end, ratio in durationParameters:
            percentChange = 1 + (ratio - 1) * stepAmount
            durationPointList.append((start, percentChange))
            durationPointList.append((end, percentChange))
        
        outputPrefix = "%s_%0.3g" % (outputName, stepAmount)
        durationTierFN = join(durationTierPath,
                              "%s.DurationTier" % outputPrefix)
        outputWavFN = join(outputPath, "%s.wav" % outputPrefix)
        durationTier = dataio.PointObject2D(durationPointList, dataio.DURATION,
                                            0, fromWavDuration)
        durationTier.save(durationTierFN)
        
        praat_scripts.resynthesizeDuration(praatEXE,
                                           fromWavFN,
                                           durationTierFN,
                                           outputWavFN,
                                           outputMinPitch, outputMaxPitch)