def _find_contpix_given_cuts(f_cut, sig_cut, wl, fluxes, ivars):
    """ Find and return continuum pixels given the flux and sigma cut

    Parameters
    ----------
    f_cut: float
        the upper limit imposed on the quantity (fbar-1)
    sig_cut: float
        the upper limit imposed on the quantity (f_sig)
    wl: numpy ndarray of length npixels
        rest-frame wavelength vector
    fluxes: numpy ndarray of shape (nstars, npixels)
        pixel intensities
    ivars: numpy ndarray of shape nstars, npixels
        inverse variances, parallel to fluxes

    Returns
    -------
    contmask: boolean mask of length npixels
        True indicates that the pixel is continuum
    """
    f_bar = np.median(fluxes, axis=0)
    sigma_f = np.var(fluxes, axis=0)
    bad = np.logical_and(f_bar==0, sigma_f==0)
    cont1 = np.abs(f_bar-1) <= f_cut
    cont2 = sigma_f <= sig_cut
    contmask = np.logical_and(cont1, cont2)
    contmask[bad] = False
    return contmask