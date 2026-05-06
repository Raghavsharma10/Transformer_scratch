def _find_contpix_regions(wl, fluxes, ivars, frac, ranges):
    """ Find continuum pix in a spectrum split into chunks

    Parameters
    ----------
    wl: numpy ndarray
        rest-frame wavelength vector

    fluxes: numpy ndarray
        pixel intensities

    ivars: numpy ndarray
        inverse variances, parallel to fluxes

    frac: float
        fraction of pixels in spectrum to be found as continuum

    ranges: list, array
        starts and ends indicating location of chunks in array

    Returns
    ------
    contmask: numpy ndarray, boolean
        True indicates continuum pixel
    """
    contmask = np.zeros(len(wl), dtype=bool)
    for chunk in ranges:
        start = chunk[0]
        stop = chunk[1]
        contmask[start:stop] = _find_contpix(
                wl[start:stop], fluxes[:,start:stop], ivars[:,start:stop], frac)
    return contmask