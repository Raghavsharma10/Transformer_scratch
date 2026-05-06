def getLinearityFunction(expTimes, imgs, mxIntensity=65535, min_ascent=0.001,
                         ):
    '''
    returns offset, ascent 
    of image(expTime) = offset + ascent*expTime
    '''
    # TODO: calculate [min_ascent] from noise function
    # instead of having it as variable

    ascent, offset, error = linRegressUsingMasked2dArrays(
        expTimes, imgs, imgs > mxIntensity)

    ascent[np.isnan(ascent)] = 0
    # remove low frequent noise:
    if min_ascent > 0:
        i = ascent < min_ascent
        offset[i] += (0.5 * (np.min(expTimes) + np.max(expTimes))) * ascent[i]
        ascent[i] = 0

    return offset, ascent, error