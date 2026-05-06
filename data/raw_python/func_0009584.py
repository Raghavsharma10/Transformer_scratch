def scaleSignal(img, fitParams=None,
                backgroundToZero=False, reference=None):
    '''
    scale the image between...

    backgroundToZero=True -> 0 (average background) and 1 (maximum signal)
    backgroundToZero=False -> signal+-3std

    reference -> reference image -- scale image to fit this one

    returns:
    scaled image
    '''
    img = imread(img)
    if reference is not None:
        #         def fn(ii, m,n):
        #             return ii*m+n
        #         curve_fit(fn, img[::10,::10], ref[::10,::10])

        low, high = signalRange(img, fitParams)
        low2, high2 = signalRange(reference)
        img = np.asfarray(img)
        ampl = (high2 - low2) / (high - low)
        img -= low
        img *= ampl
        img += low2
        return img
    else:
        offs, div = scaleParams(img, fitParams, backgroundToZero)
        img = np.asfarray(img) - offs
        img /= div
        print('offset: %s, divident: %s' % (offs, div))
        return img