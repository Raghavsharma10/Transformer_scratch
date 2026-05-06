def getBackgroundRange(fitParams):
    '''
    return minimum, average, maximum of the background peak
    '''
    smn, _, _ = getSignalParameters(fitParams)

    bg = fitParams[0]
    _, avg, std = bg
    bgmn = max(0, avg - 3 * std)

    if avg + 4 * std < smn:
        bgmx = avg + 4 * std
    if avg + 3 * std < smn:
        bgmx = avg + 3 * std
    if avg + 2 * std < smn:
        bgmx = avg + 2 * std
    else:
        bgmx = avg + std
    return bgmn, avg, bgmx