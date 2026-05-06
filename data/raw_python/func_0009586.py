def hasBackground(fitParams):
    '''
    compare the height of putative bg and signal peak
    if ratio if too height assume there is no background
    '''
    signal = getSignalPeak(fitParams)
    bg = getBackgroundPeak(fitParams)
    if signal == bg:
        return False
    r = signal[0] / bg[0]
    if r < 1:
        r = 1 / r
    return r < 100