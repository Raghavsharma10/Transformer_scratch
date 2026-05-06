def get_lt(hdr):
    """Obtain the LTV and LTM keyword values.

    Note that this returns the values just as read from the header,
    which means in particular that the LTV values are for one-indexed
    pixel coordinates.

    LTM keywords are the diagonal elements of MWCS linear
    transformation matrix, while LTV's are MWCS linear transformation
    vector (1-indexed).

    .. note:: Translated from ``calacs/lib/getlt.c``.

    Parameters
    ----------
    hdr : obj
        Extension header.

    Returns
    -------
    ltm, ltv : tuple of float
        ``(LTM1_1, LTM2_2)`` and ``(LTV1, LTV2)``.
        Values are ``(1, 1)`` and ``(0, 0)`` if not found,
        to accomodate reference files with missing info.

    Raises
    ------
    ValueError
        Invalid LTM* values.

    """
    ltm = (hdr.get('LTM1_1', 1.0), hdr.get('LTM2_2', 1.0))

    if ltm[0] <= 0 or ltm[1] <= 0:
        raise ValueError('(LTM1_1, LTM2_2) = {0} is invalid'.format(ltm))

    ltv = (hdr.get('LTV1', 0.0), hdr.get('LTV2', 0.0))
    return ltm, ltv