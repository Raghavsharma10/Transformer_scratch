def _cont_norm_regions(fluxes, ivars, cont, ranges):
    """ Perform continuum normalization for spectra in chunks

    Useful for spectra that have gaps

    Parameters
    ---------
    fluxes: numpy ndarray
        pixel intensities
    ivars: numpy ndarray
        inverse variances, parallel to fluxes
    cont: numpy ndarray
        the continuum
    ranges: list or np ndarray
        the chunks that the spectrum should be split into

    Returns
    -------
    norm_fluxes: numpy ndarray
        normalized pixel intensities
    norm_ivars: numpy ndarray
        rescaled inverse variances
    """
    nstars = fluxes.shape[0]
    norm_fluxes = np.zeros(fluxes.shape)
    norm_ivars = np.zeros(ivars.shape)
    for chunk in ranges:
        start = chunk[0]
        stop = chunk[1]
        output = _cont_norm(fluxes[:,start:stop],
                           ivars[:,start:stop],
                           cont[:,start:stop])
        norm_fluxes[:,start:stop] = output[0]
        norm_ivars[:,start:stop] = output[1]
    for jj in range(nstars):
        bad = (norm_ivars[jj,:] == 0.)
        norm_fluxes[jj,:][bad] = 1.
    return norm_fluxes, norm_ivars