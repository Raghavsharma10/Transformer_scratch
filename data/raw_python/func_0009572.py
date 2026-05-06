def getDarkCurrentFunction(exposuretimes, imgs, **kwargs):
    '''
    get dark current function from given images and exposure times
    '''
    exposuretimes, imgs = getDarkCurrentAverages(exposuretimes, imgs)
    offs, ascent, rmse = getLinearityFunction(exposuretimes, imgs, **kwargs)
    return offs, ascent, rmse