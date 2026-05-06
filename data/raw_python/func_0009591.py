def _equalizeHistogram(img):
    '''
    histogram equalisation not bounded to int() or an image depth of 8 bit
    works also with negative numbers
    '''
    # to float if int:
    intType = None
    if 'f' not in img.dtype.str:
        TO_FLOAT_TYPES = {np.dtype('uint8'): np.float16,
                          np.dtype('uint16'): np.float32,
                          np.dtype('uint32'): np.float64,
                          np.dtype('uint64'): np.float64}

        intType = img.dtype
        img = img.astype(TO_FLOAT_TYPES[intType], copy=False)

    # get image deph
    DEPTH_TO_NBINS = {np.dtype('float16'): 256,  # uint8
                      np.dtype('float32'): 32768,  # uint16
                      np.dtype('float64'): 2147483648}  # uint32
    nBins = DEPTH_TO_NBINS[img.dtype]

    # scale to -1 to 1 due to skikit-image restrictions
    mn, mx = np.amin(img), np.amax(img)
    if abs(mn) > abs(mx):
        mx = mn
    img /= mx
    img = exposure.equalize_hist(img, nbins=nBins)

    img *= mx

    if intType:
        img = img.astype(intType)
    return img