def _find_contpix(wl, fluxes, ivars, target_frac):
    """ Find continuum pix in spec, meeting a set target fraction

    Parameters
    ----------
    wl: numpy ndarray
        rest-frame wavelength vector

    fluxes: numpy ndarray
        pixel intensities
    
    ivars: numpy ndarray
        inverse variances, parallel to fluxes

    target_frac: float
        the fraction of pixels in spectrum desired to be continuum

    Returns
    -------
    contmask: boolean numpy ndarray
        True corresponds to continuum pixels
    """
    print("Target frac: %s" %(target_frac))
    bad1 = np.median(ivars, axis=0) == SMALL
    bad2 = np.var(ivars, axis=0) == 0
    bad = np.logical_and(bad1, bad2)
    npixels = len(wl)-sum(bad)
    f_cut = 0.0001
    stepsize = 0.0001
    sig_cut = 0.0001
    contmask = _find_contpix_given_cuts(f_cut, sig_cut, wl, fluxes, ivars)
    if npixels > 0:
        frac = sum(contmask)/float(npixels)
    else:
        frac = 0
    while (frac < target_frac): 
        f_cut += stepsize
        sig_cut += stepsize
        contmask = _find_contpix_given_cuts(f_cut, sig_cut, wl, fluxes, ivars)
        if npixels > 0:
            frac = sum(contmask)/float(npixels)
        else:
            frac = 0
    if frac > 0.10*npixels:
        print("Warning: Over 10% of pixels identified as continuum.")
    print("%s out of %s pixels identified as continuum" %(sum(contmask), 
                                                          npixels))
    print("Cuts: f_cut %s, sig_cut %s" %(f_cut, sig_cut))
    return contmask