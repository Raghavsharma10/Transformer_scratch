def wave_range(bins, cenwave, npix, mode='round'):
    """Calculate the wavelength range covered by the given number of pixels
    centered on the given central wavelength of the given bins.

    Parameters
    ----------
    bins : array-like
        Wavelengths at bin centers, each centered on a pixel.
        Must be 1D array.

    cenwave : float
        Desired central wavelength, in the same unit as ``bins``.

    npix : int
        Desired number of pixels, centered on ``cenwave``.

    mode : {'round', 'min', 'max', 'none'}
        Determines how the pixels at the edges of the wavelength range
        are handled. All the options, except 'none', will return
        wavelength range edges that correspond to pixel edges:

            * 'round' - Wavelength range edges are the pixel edges
              and the range spans exactly ``npix`` pixels. An edge
              that falls in the center of a bin is rounded to the
              nearest pixel edge. This is the default.

            * 'min' - Wavelength range is shrunk such that it includes
              an integer number of pixels and its edges fall on pixel
              edges. It may not span exactly ``npix`` pixels.

            * 'max' - Wavelength range is expanded such that it
              includes an integer number of pixels and its edges fall
              on pixel edges. It may not span exactly ``npix`` pixels.

            * 'none' - Exact wavelength range is returned. The edges
              may not correspond to pixel edges, but it covers exactly
              ``npix`` pixels.

    Returns
    -------
    wave1, wave2 : float
        Lower and upper limits of the wavelength range.

    Raises
    ------
    synphot.exceptions.OverlapError
        Given central wavelength is not within the given bins
        or the wavelength range would exceed the bin limits.

    synphot.exceptions.SynphotError
        Invalid inputs or calculation failed.

    """
    mode = mode.lower()

    if mode not in ('round', 'min', 'max', 'none'):
        raise exceptions.SynphotError(
            'mode={0} is invalid, must be "round", "min", "max", '
            'or "none".'.format(mode))

    if not isinstance(npix, int):
        raise exceptions.SynphotError('npix={0} is invalid.'.format(npix))

    # Bin values must be in ascending order.
    if bins[0] > bins[-1]:
        bins = bins[::-1]

    # Central wavelength must be within given bins.
    if cenwave < bins[0] or cenwave > bins[-1]:
        raise exceptions.OverlapError(
            'cenwave={0} is not within binset (min={1}, max={2}).'.format(
                cenwave, bins[0], bins[-1]))

    # Find the index the central wavelength among bins
    diff = cenwave - bins
    ind = np.argmin(np.abs(diff))

    # Calculate fractional index
    frac_ind = float(ind)
    if diff[ind] < 0:
        frac_ind += diff[ind] / (bins[ind] - bins[ind - 1])
    elif diff[ind] > 0:
        frac_ind += diff[ind] / (bins[ind + 1] - bins[ind])

    # Calculate fractional indices of the edges
    half_npix = npix / 2.0
    frac_ind1 = frac_ind - half_npix
    frac_ind2 = frac_ind + half_npix

    # Calculated edges must not exceed bin edges
    if frac_ind1 < -0.5:
        raise exceptions.OverlapError(
            'Lower limit of wavelength range is out of bounds.')
    if frac_ind2 > (bins.size - 0.5):
        raise exceptions.OverlapError(
            'Upper limit of wavelength range is out of bounds.')

    frac1, int1 = np.modf(frac_ind1)
    frac2, int2 = np.modf(frac_ind2)
    int1 = int(int1)
    int2 = int(int2)

    if mode == 'round':
        # Lower end of wavelength range
        if frac1 >= 0:
            # end is somewhere greater than binset[0] so we can just
            # interpolate between two neighboring values going with upper edge
            wave1 = bins[int1:int1 + 2].mean()
        else:
            # end is below the lowest binset value, but not by enough to
            # trigger an exception
            wave1 = bins[0] - (bins[0:2].mean() - bins[0])

        # Upper end of wavelength range
        if int2 < bins.shape[0] - 1:
            # end is somewhere below binset[-1] so we can just interpolate
            # between two neighboring values, going with the upper edge.
            wave2 = bins[int2:int2 + 2].mean()
        else:
            # end is above highest binset value but not by enough to
            # trigger an exception
            wave2 = bins[-1] + (bins[-1] - bins[-2:].mean())

    elif mode == 'min':
        # Lower end of wavelength range
        if frac1 <= 0.5 and int1 < bins.shape[0] - 1:
            # not at the lowest possible edge and pixel i included
            wave1 = bins[int1:int1 + 2].mean()
        elif frac1 > 0.5 and int1 < bins.shape[0] - 2:
            # not at the lowest possible edge and pixel i not included
            wave1 = bins[int1 + 1:int1 + 3].mean()
        elif frac1 == -0.5:
            # at the lowest possible edge
            wave1 = bins[0] - (bins[0:2].mean() - bins[0])
        else:  # pragma: no cover
            raise exceptions.SynphotError(
                'mode={0} gets unexpected frac1={1}, int1={2}'.format(
                    mode, frac1, int1))

        # Upper end of wavelength range
        if frac2 >= 0.5 and int2 < bins.shape[0] - 1:
            # not out at the end and pixel i included
            wave2 = bins[int2:int2 + 2].mean()
        elif frac2 < 0.5 and int2 < bins.shape[0]:
            # not out at end and pixel i not included
            wave2 = bins[int2 - 1:int2 + 1].mean()
        elif frac2 == 0.5 and int2 == bins.shape[0] - 1:
            # at the very end
            wave2 = bins[-1] + (bins[-1] - bins[-2:].mean())
        else:  # pragma: no cover
            raise exceptions.SynphotError(
                'mode={0} gets unexpected frac2={1}, int2={2}'.format(
                    mode, frac2, int2))

    elif mode == 'max':
        # Lower end of wavelength range
        if frac1 < 0.5 and int1 < bins.shape[0]:
            # not at the lowest possible edge and pixel i included
            wave1 = bins[int1 - 1:int1 + 1].mean()
        elif frac1 >= 0.5 and int1 < bins.shape[0] - 1:
            # not at the lowest possible edge and pixel i not included
            wave1 = bins[int1:int1 + 2].mean()
        elif frac1 == -0.5:
            # at the lowest possible edge
            wave1 = bins[0] - (bins[0:2].mean() - bins[0])
        else:  # pragma: no cover
            raise exceptions.SynphotError(
                'mode={0} gets unexpected frac1={1}, int1={2}'.format(
                    mode, frac1, int1))

        # Upper end of wavelength range
        if frac2 > 0.5 and int2 < bins.shape[0] - 2:
            # not out at the end and pixel i included
            wave2 = bins[int2 + 1:int2 + 3].mean()
        elif frac2 <= 0.5 and int2 < bins.shape[0] - 1:
            # not out at end and pixel i not included
            wave2 = bins[int2:int2 + 2].mean()
        elif frac2 == 0.5 and int2 == bins.shape[0] - 1:
            # at the very end
            wave2 = bins[-1] + (bins[-1] - bins[-2:].mean())
        else:  # pragma: no cover
            raise exceptions.SynphotError(
                'mode={0} gets unexpected frac2={1}, int2={2}'.format(
                    mode, frac2, int2))

    else:  # mode == 'none'
        wave1 = bins[int1] + frac1 * (bins[int1 + 1] - bins[int1])
        wave2 = bins[int2] + frac2 * (bins[int2 + 1] - bins[int2])

    return wave1, wave2