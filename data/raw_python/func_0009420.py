def standardDeviation2d(img, ksize=5, blurred=None):
    '''
    calculate the spatial resolved standard deviation 
    for a given 2d array
    
    ksize   -> kernel size
    
    blurred(optional) -> with same ksize gaussian filtered image
                      setting this parameter reduces processing time
    '''
    if ksize not in (list, tuple):
        ksize = (ksize,ksize)

    if blurred is None:
        blurred = gaussian_filter(img, ksize)
    else:
        assert blurred.shape == img.shape
    
    std = np.empty_like(img)
    
    _calc(img, ksize[0], ksize[1], blurred, std)
    
    return std