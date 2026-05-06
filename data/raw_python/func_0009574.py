def positionToIntensityUncertainty(image, sx, sy, kernelSize=None):
    '''
    calculates the estimated standard deviation map from the changes
    of neighbouring pixels from a center pixel within a point spread function
    defined by a std.dev. in x and y taken from the (sx, sy) maps 

    sx,sy -> either 2d array of same shape as [image]
             of single values
    '''
    psf_is_const = not isinstance(sx, np.ndarray)
    if not psf_is_const:
        assert image.shape == sx.shape == sy.shape, \
            "Image and position uncertainty maps need to have same size"
        if kernelSize is None:
            kernelSize = _kSizeFromStd(max(sx.max(), sy.max()))
    else:
        assert type(sx) in (int, float) and type(sx) in (int, float), \
            "Image and position uncertainty values need to be int OR float"
        if kernelSize is None:
            kernelSize = _kSizeFromStd(max(sx, sy))

    if image.dtype.kind == 'u':
        image = image.astype(int)  # otherwise stack overflow through uint
    size = kernelSize // 2
    if size < 1:
        size = 1
    kernelSize = 1 + 2 * size
    # array to be filled by individual psf of every pixel:
    psf = np.zeros((kernelSize, kernelSize))
    # intensity uncertainty as stdev:
    sint = np.zeros(image.shape)
    if psf_is_const:
        _calc_constPSF(image, sint, sx, sy, psf, size)
    else:
        _calc_variPSF(image, sint, sx, sy, psf, size)
    return sint