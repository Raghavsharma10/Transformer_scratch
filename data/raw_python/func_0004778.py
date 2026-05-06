def _find_cont_running_quantile(wl, fluxes, ivars, q, delta_lambda,
                                verbose=False):
    """ Perform continuum normalization using a running quantile

    Parameters
    ----------
    wl: numpy ndarray 
        wavelength vector
    fluxes: numpy ndarray of shape (nstars, npixels)
        pixel intensities
    ivars: numpy ndarray of shape (nstars, npixels)
        inverse variances, parallel to fluxes
    q: float
        the desired quantile cut
    delta_lambda: int
        the number of pixels over which the median is calculated

    Output
    ------
    norm_fluxes: numpy ndarray of shape (nstars, npixels)
        normalized pixel intensities
    norm_ivars: numpy ndarray of shape (nstars, npixels)
        rescaled pixel invariances
    """
    cont = np.zeros(fluxes.shape)
    nstars = fluxes.shape[0]
    for jj in range(nstars):
        if verbose:
            print("cont_norm_q(): working on star [%s/%s]..." % (jj+1, nstars))
        flux = fluxes[jj,:]
        ivar = ivars[jj,:]
        for ll, lam in enumerate(wl):
            indx = (np.where(abs(wl-lam) < delta_lambda))[0]
            flux_cut = flux[indx]
            ivar_cut = ivar[indx]
            cont[jj, ll] = _weighted_median(flux_cut, ivar_cut, q)
    return cont