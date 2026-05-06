def get_corner(hdr, rsize=1):
    """Obtain bin and corner information for a subarray.

    ``LTV1``, ``LTV2``, ``LTM1_1``, and ``LTM2_2`` keywords
    are extracted from the given extension header and converted
    to bin and corner values (0-indexed).

    ``LTV1`` for the CCD uses the beginning of the illuminated
    portion as the origin, not the beginning of the overscan region.
    Thus, the computed X-corner has the same origin as ``LTV1``,
    which is what we want, but it differs from the ``CENTERA1``
    header keyword, which has the beginning of the overscan region
    as origin.

    .. note:: Translated from ``calacs/lib/getcorner.c``.

    Parameters
    ----------
    hdr : obj
        Extension header.

    rsize : int, optional
        Size of reference pixel in units of high-res pixels.

    Returns
    -------
    bin : tuple of int
        Pixel size in X and Y.

    corner : tuple of int
        Corner of subarray in X and Y.

    """
    ltm, ltv = get_lt(hdr)
    return from_lt(rsize, ltm, ltv)