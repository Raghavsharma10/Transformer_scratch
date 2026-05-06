def scaleSignalCut(img, ratio, nbins=100):
    '''
    scaling img cutting x percent of top and bottom part of histogram
    '''
    start, stop = scaleSignalCutParams(img, ratio, nbins)
    img = img - start
    img /= (stop - start)
    return img