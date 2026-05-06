def medianThreshold(img, threshold=0.1, size=3, condition='>', copy=True):
    '''
    set every the pixel value of the given [img] to the median filtered one
    of a given kernel [size]
    in case the relative [threshold] is exeeded
    condition = '>' OR '<'
    '''
    from scipy.ndimage import median_filter

    indices = None
    if threshold > 0:
        blur = np.asfarray(median_filter(img, size=size))
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):

            if condition == '>':
                indices = abs((img - blur) / blur) > threshold
            else:
                indices = abs((img - blur) / blur) < threshold

        if copy:
            img = img.copy()

        img[indices] = blur[indices]
    return img, indices