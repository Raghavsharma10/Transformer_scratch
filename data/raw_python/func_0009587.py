def signalMinimum2(img, bins=None):
    '''
    minimum position between signal and background peak
    '''
    f = FitHistogramPeaks(img, bins=bins)
    i = signalPeakIndex(f.fitParams)
    spos = f.fitParams[i][1]
#     spos = getSignalPeak(f.fitParams)[1]
#     bpos = getBackgroundPeak(f.fitParams)[1]
    bpos = f.fitParams[i - 1][1]
    ind = np.logical_and(f.xvals > bpos, f.xvals < spos)
    try:
        i = np.argmin(f.yvals[ind])
        return f.xvals[ind][i]
    except ValueError as e:
        if bins is None:
            return signalMinimum2(img, bins=400)
        else:
            raise e