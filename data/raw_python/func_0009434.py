def calcNLF(img, img2=None, signal=None, mn_mx_nbins=None, x=None,
            averageFn='AAD',
            signalFromMultipleImages=False):
    '''
    Calculate the noise level function (NLF) as f(intensity)
    using one or two image.
    The approach for this work is published in JPV##########

    img2 - 2nd image taken under same conditions
           used to estimate noise via image difference

    signalFromMultipleImages - whether the signal is an average of multiple
        images and not just got from one median filtered image
    '''
    # CONSTANTS:
    # factor Root mead square to average-absolute-difference:
    F_RMS2AAD = (2 / np.pi)**-0.5
    F_NOISE_WITH_MEDIAN = 1 + (1 / 3**2)
    N_BINS = 100
    MEDIAN_KERNEL_SIZE = 3

    def _averageAbsoluteDeviation(d):
        return np.mean(np.abs(d)) * F_RMS2AAD

    def _rootMeanSquare(d):
        return (d**2).mean()**0.5

    if averageFn == 'AAD':
        averageFn = _averageAbsoluteDeviation
    else:
        averageFn = _rootMeanSquare

    img = np.asfarray(img)

    if img2 is None:
        if signal is None:
            signal = median_filter(img, MEDIAN_KERNEL_SIZE)
        if signalFromMultipleImages:
            diff = img - signal
        else:
            # difference between the filtered and original image:
            diff = (img - signal) * F_NOISE_WITH_MEDIAN

    else:
        img2 = np.asfarray(img2)
        diff = (img - img2)
        # 2**0.5 because noise is subtracted by noise
        # and variance of sum = sum of variance:
        # var(immg1-img2)~2*var(img)
        # std(2*var) = 2**0.5*var**0.5
        diff /= 2**0.5
        if signal is None:
            signal = median_filter(0.5 * (img + img2), MEDIAN_KERNEL_SIZE)

    if mn_mx_nbins is not None:
        mn, mx, nbins = mn_mx_nbins
        min_len = 0
    else:
        mn, mx = _getMinMax(signal)
        s = img.shape
        min_len = int(s[0] * s[1] * 1e-3)
        if min_len < 1:
            min_len = 5
        # number of bins/different intensity ranges to analyse:
        nbins = N_BINS
        if mx - mn < nbins:
            nbins = int(mx - mn)
    # bin width:
    step = (mx - mn) / nbins

    # empty arrays:
    y = np.empty(shape=nbins)
    set_x = False
    if x is None:
        set_x = True
        x = np.empty(shape=nbins)
    # give bins with more samples more weight:
    weights = np.zeros(shape=nbins)

    # cur step:
    m = mn
    for n in range(nbins):
        # get indices of all pixel with in a bin:
        ind = np.logical_and(signal >= m, signal <= m + step)
        m += step
        d = diff[ind]
        ld = len(d)
        if ld >= min_len:
            weights[n] = ld
            # average absolute deviation (AAD),
            # scaled to RMS:
            y[n] = averageFn(d)
            if set_x:
                x[n] = m - 0.5 * step

    return x, y, weights, signal