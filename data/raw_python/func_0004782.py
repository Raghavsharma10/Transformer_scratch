def _cont_norm(fluxes, ivars, cont):
    """ Continuum-normalize a continuous segment of spectra.

    Parameters
    ----------
    fluxes: numpy ndarray 
        pixel intensities
    ivars: numpy ndarray 
        inverse variances, parallel to fluxes
    contmask: boolean mask
        True indicates that pixel is continuum

    Returns
    -------
    norm_fluxes: numpy ndarray
        normalized pixel intensities
    norm_ivars: numpy ndarray
        rescaled inverse variances
    """
    nstars = fluxes.shape[0]
    npixels = fluxes.shape[1]
    norm_fluxes = np.ones(fluxes.shape)
    norm_ivars = np.zeros(ivars.shape)
    bad = cont == 0.
    norm_fluxes = np.ones(fluxes.shape)
    norm_fluxes[~bad] = fluxes[~bad] / cont[~bad]
    norm_ivars = cont**2 * ivars
    return norm_fluxes, norm_ivars