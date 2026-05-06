def getSignalParameters(fitParams, n_std=3):
    '''
    return minimum, average, maximum of the signal peak
    '''
    signal = getSignalPeak(fitParams)
    mx = signal[1] + n_std * signal[2]
    mn = signal[1] - n_std * signal[2]
    if mn < fitParams[0][1]:
        mn = fitParams[0][1]  # set to bg
    return mn, signal[1], mx