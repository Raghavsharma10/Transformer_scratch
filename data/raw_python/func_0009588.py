def signalMinimum(img, fitParams=None, n_std=3):
    '''
    intersection between signal and background peak
    '''
    if fitParams is None:
        fitParams = FitHistogramPeaks(img).fitParams
    assert len(fitParams) > 1, 'need 2 peaks so get minimum signal'

    i = signalPeakIndex(fitParams)
    signal = fitParams[i]
    bg = getBackgroundPeak(fitParams)
    smn = signal[1] - n_std * signal[2]
    bmx = bg[1] + n_std * bg[2]
    if smn > bmx:
        return smn
    # peaks are overlapping
    # define signal min. as intersection between both Gaussians

    def solve(p1, p2):
        s1, m1, std1 = p1
        s2, m2, std2 = p2
        a = (1 / (2 * std1**2)) - (1 / (2 * std2**2))
        b = (m2 / (std2**2)) - (m1 / (std1**2))
        c = (m1**2 / (2 * std1**2)) - (m2**2 / (2 * std2**2)) - \
            np.log(((std2 * s1) / (std1 * s2)))
        return np.roots([a, b, c])
    i = solve(bg, signal)
    try:
        return i[np.logical_and(i > bg[1], i < signal[1])][0]
    except IndexError:
        # this error shouldn't occur... well
        return max(smn, bmx)