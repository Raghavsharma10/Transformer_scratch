def pixel_range(bins, waverange, mode='round'):
    """Calculate the number of pixels within the given wavelength range
    and the given bins.

    Parameters
    ----------
    bins : array-like
        Wavelengths at bin centers, each centered on a pixel.
        Must be 1D array.

    waverange : tuple of float
        Lower and upper limits of the desired wavelength range,
        in the same unit as ``bins``.

    mode : {'round', 'min', 'max', 'none'}
        Determines how the pixels at the edges of the wavelength range
        are handled. All the options, except 'none', will return
        an integer number of pixels:

            * 'round' - Wavelength range edges that fall in the middle
              of a pixel are counted if more than half of the pixel is
              within the given wavelength range. Edges that fall in
              the center of a pixel are rounded to the nearest pixel
              edge. This is the default.

            * 'min' - Only pixels wholly within the given wavelength
              range are counted.

            * 'max' - Pixels that are within the given wavelength range
              by any margin are counted.

            * 'none' - The exact number of encompassed pixels,
              including fractional pixels, is returned.

    Returns
    -------
    npix : number
        Number of pixels.

    Raises
    ------
    synphot.exceptions.OverlapError
        Given wavelength range exceeds the bounds of given bins.

    synphot.exceptions.SynphotError
        Invalid mode.

    """
    mode = mode.lower()

    if mode not in ('round', 'min', 'max', 'none'):
        raise exceptions.SynphotError(
            'mode={0} is invalid, must be "round", "min", "max", '
            'or "none".'.format(mode))

    if waverange[0] < waverange[-1]:
        wave1 = waverange[0]
        wave2 = waverange[-1]
    else:
        wave1 = waverange[-1]
        wave2 = waverange[0]

    # Bin values must be in ascending order.
    if bins[0] > bins[-1]:
        bins = bins[::-1]

    # Wavelength range must be within bins
    minwave = bins[0] - (bins[0:2].mean() - bins[0])
    maxwave = bins[-1] + (bins[-1] - bins[-2:].mean())
    if wave1 < minwave or wave2 > maxwave:
        raise exceptions.OverlapError(
            'Wavelength range ({0}, {1}) is out of bounds of bins '
            '(min={2}, max={3}).'.format(wave1, wave2, minwave, maxwave))

    if wave1 == wave2:
        return 0

    if mode == 'round':
        ind1 = bins.searchsorted(wave1, side='right')
        ind2 = bins.searchsorted(wave2, side='right')
    else:
        ind1 = bins.searchsorted(wave1, side='left')
        ind2 = bins.searchsorted(wave2, side='left')

    if mode == 'round':
        npix = ind2 - ind1

    elif mode == 'min':
        # for ind1, figure out if pixel ind1 is wholly included or not.
        # do this by figuring out where wave1 is between ind1 and ind1-1.
        frac = (bins[ind1] - wave1) / (bins[ind1] - bins[ind1 - 1])
        if frac < 0.5:
            # ind1 is only partially included
            ind1 += 1

        # similar but reversed procedure for ind2
        frac = (wave2 - bins[ind2 - 1]) / (bins[ind2] - bins[ind2 - 1])
        if frac < 0.5:
            # ind2 is only partially included
            ind2 -= 1

        npix = ind2 - ind1

    elif mode == 'max':
        # for ind1, figure out if pixel ind1-1 is partially included or not.
        # do this by figuring out where wave1 is between ind1 and ind1-1.
        frac = (wave1 - bins[ind1 - 1]) / (bins[ind1] - bins[ind1 - 1])
        if frac < 0.5:
            # ind1 is partially included
            ind1 -= 1

        # similar but reversed procedure for ind2
        frac = (bins[ind2] - wave2) / (bins[ind2] - bins[ind2 - 1])
        if frac < 0.5:
            # ind2 is partially included
            ind2 += 1

        npix = ind2 - ind1

    else:  # mode == 'none'
        # calculate fractional indices
        frac1 = ind1 - (bins[ind1] - wave1) / (bins[ind1] - bins[ind1 - 1])
        frac2 = ind2 - (bins[ind2] - wave2) / (bins[ind2] - bins[ind2 - 1])
        npix = frac2 - frac1

    return npix