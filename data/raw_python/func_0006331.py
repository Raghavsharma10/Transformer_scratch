def invert_pixel_mask(mask):
    '''Invert pixel mask (0->1, 1(and greater)->0).

    Parameters
    ----------
    mask : array-like
        Mask.

    Returns
    -------
    inverted_mask : array-like
        Inverted Mask.
    '''
    inverted_mask = np.ones(shape=(80, 336), dtype=np.dtype('>u1'))
    inverted_mask[mask >= 1] = 0
    return inverted_mask