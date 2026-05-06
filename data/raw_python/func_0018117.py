def from_lt(rsize, ltm, ltv):
    """Compute the corner location and pixel size in units
    of unbinned pixels.

    .. note:: Translated from ``calacs/lib/fromlt.c``.

    Parameters
    ----------
    rsize : int
        Reference pixel size. Usually 1.

    ltm, ltv : tuple of float
        See :func:`get_lt`.

    Returns
    -------
    bin : tuple of int
        Pixel size in X and Y.

    corner : tuple of int
        Corner of subarray in X and Y.

    """
    dbinx = rsize / ltm[0]
    dbiny = rsize / ltm[1]

    dxcorner = (dbinx - rsize) - dbinx * ltv[0]
    dycorner = (dbiny - rsize) - dbiny * ltv[1]

    # Round off to the nearest integer.
    bin = (_nint(dbinx), _nint(dbiny))
    corner = (_nint(dxcorner), _nint(dycorner))

    return bin, corner