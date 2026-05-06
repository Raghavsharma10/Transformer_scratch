def load_spatial_filters(packed=True):
    """Load spatial-filters kernel

    Parameters
    ----------
    packed : bool
        Whether or not the data should be in "packed" representation
        for use in GLSL code.

    Returns
    -------
    kernel : array
        16x1024x4 (packed float in rgba) or
        16x1024 (unpacked float)
        16 interpolation kernel with length 1024 each.

    names : tuple of strings
        Respective interpolation names, plus "Nearest" which does
        not require a filter but can still be used
    """
    names = ("Bilinear", "Hanning", "Hamming", "Hermite",
             "Kaiser", "Quadric", "Bicubic", "CatRom",
             "Mitchell", "Spline16", "Spline36", "Gaussian",
             "Bessel", "Sinc", "Lanczos", "Blackman", "Nearest")

    kernel = np.load(op.join(DATA_DIR, 'spatial-filters.npy'))
    if packed:
        # convert the kernel to a packed representation
        kernel = pack_unit(kernel)

    return kernel, names