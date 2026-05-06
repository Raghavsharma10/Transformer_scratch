def find_line(scihdu, refhdu):
    """Obtain bin factors and corner location to extract
    and bin the appropriate subset of a reference image to
    match a science image.

    If the science image has zero offset and is the same size and
    binning as the reference image, ``same_size`` will be set to
    `True`. Otherwise, the values of ``rx``, ``ry``, ``x0``, and
    ``y0`` will be assigned.

    Normally the science image will be binned the same or more
    than the reference image. In that case, ``rx`` and ``ry``
    will be the bin size of the science image divided by the
    bin size of the reference image.

    If the binning of the reference image is greater than the
    binning of the science image, the ratios (``rx`` and ``ry``)
    of the bin sizes will be the reference image size divided by
    the science image bin size. This is not necessarily an error.

    .. note:: Translated from ``calacs/lib/findbin.c``.

    Parameters
    ----------
    scihdu, refhdu : obj
        Extension HDU's of the science and reference image,
        respectively.

    Returns
    -------
    same_size : bool
        `True` if zero offset and same size and binning.

    rx, ry : int
        Ratio of bin sizes.

    x0, y0 : int
        Location of start of subimage in reference image.

    Raises
    ------
    ValueError
        Science and reference data size mismatch.

    """
    sci_bin, sci_corner = get_corner(scihdu.header)
    ref_bin, ref_corner = get_corner(refhdu.header)

    # We can use the reference image directly, without binning
    # and without extracting a subset.
    if (sci_corner[0] == ref_corner[0] and sci_corner[1] == ref_corner[1] and
            sci_bin[0] == ref_bin[0] and sci_bin[1] == ref_bin[1] and
            scihdu.data.shape[1] == refhdu.data.shape[1]):
        same_size = True
        rx = 1
        ry = 1
        x0 = 0
        y0 = 0

    # Reference image is binned more than the science image.
    elif ref_bin[0] > sci_bin[0] or ref_bin[1] > sci_bin[1]:
        same_size = False
        rx = ref_bin[0] / sci_bin[0]
        ry = ref_bin[1] / sci_bin[1]
        x0 = (sci_corner[0] - ref_corner[0]) / ref_bin[0]
        y0 = (sci_corner[1] - ref_corner[1]) / ref_bin[1]

    # For subarray input images, whether they are binned or not.
    else:
        same_size = False

        # Ratio of bin sizes.
        ratiox = sci_bin[0] / ref_bin[0]
        ratioy = sci_bin[1] / ref_bin[1]

        if (ratiox * ref_bin[0] != sci_bin[0] or
                ratioy * ref_bin[1] != sci_bin[1]):
            raise ValueError('Science and reference data size mismatch')

        # cshift is the offset in units of unbinned pixels.
        # Divide by ref_bin to convert to units of pixels in the ref image.
        cshift = (sci_corner[0] - ref_corner[0], sci_corner[1] - ref_corner[1])
        xzero = cshift[0] / ref_bin[0]
        yzero = cshift[1] / ref_bin[1]

        if (xzero * ref_bin[0] != cshift[0] or
                yzero * ref_bin[1] != cshift[1]):
            warnings.warn('Subimage offset not divisible by bin size',
                          AstropyUserWarning)

        rx = ratiox
        ry = ratioy
        x0 = xzero
        y0 = yzero

    # Ensure integer index
    x0 = int(x0)
    y0 = int(y0)

    return same_size, rx, ry, x0, y0