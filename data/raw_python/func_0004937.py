def rebinmask(mask, binx, biny, enlarge=False):
    """Re-bin (shrink or enlarge) a mask matrix.

    Inputs
    ------
    mask: np.ndarray
        mask matrix.
    binx: integer
        binning along the 0th axis
    biny: integer
        binning along the 1st axis
    enlarge: bool, optional
        direction of binning. If True, the matrix will be enlarged, otherwise
        shrinked (this is the default)

    Output
    ------
    the binned mask matrix, of shape ``M/binx`` times ``N/biny`` or ``M*binx``
    times ``N*biny``, depending on the value of ``enlarge`` (if ``mask`` is
    ``M`` times ``N`` pixels).

    Notes
    -----
    one is nonmasked, zero is masked.
    """
    if not enlarge and ((mask.shape[0] % binx) or (mask.shape[1] % biny)):
        raise ValueError(
            'The number of pixels of the mask matrix should be divisible by the binning in each direction!')
    if enlarge:
        return mask.repeat(binx, axis=0).repeat(biny, axis=1)
    else:
        return mask[::binx, ::biny]