def check_overscan(xstart, xsize, total_prescan_pixels=24,
                   total_science_pixels=4096):
    """Check image for bias columns.

    Parameters
    ----------
    xstart : int
        Starting column of the readout in detector coordinates.

    xsize : int
        Number of columns in the readout.

    total_prescan_pixels : int
        Total prescan pixels for a single amplifier on a detector.
        Default is 24 for WFC.

    total_science_pixels : int
        Total science pixels across a detector.
        Default is 4096 for WFC (across two amplifiers).

    Returns
    -------
    hasoverscan : bool
        Indication if there are bias columns in the image.

    leading : int
        Number of bias columns on the A/C amplifiers
        side of the CCDs ("TRIMX1" in ``OSCNTAB``).

    trailing : int
        Number of bias columns on the B/D amplifiers
        side of the CCDs ("TRIMX2" in ``OSCNTAB``).

    """
    hasoverscan = False
    leading = 0
    trailing = 0

    if xstart < total_prescan_pixels:
        hasoverscan = True
        leading = abs(xstart - total_prescan_pixels)

    if (xstart + xsize) > total_science_pixels:
        hasoverscan = True
        trailing = abs(total_science_pixels -
                       (xstart + xsize - total_prescan_pixels))

    return hasoverscan, leading, trailing