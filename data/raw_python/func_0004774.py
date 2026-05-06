def _find_cont_gaussian_smooth(wl, fluxes, ivars, w):
    """ Returns the weighted mean block of spectra

    Parameters
    ----------
    wl: numpy ndarray
        wavelength vector
    flux: numpy ndarray
        block of flux values 
    ivar: numpy ndarray
        block of ivar values
    L: float
        width of Gaussian used to assign weights

    Returns
    -------
    smoothed_fluxes: numpy ndarray
        block of smoothed flux values, mean spectra
    """
    print("Finding the continuum")
    bot = np.dot(ivars, w.T)
    top = np.dot(fluxes*ivars, w.T)
    bad = bot == 0
    cont = np.zeros(top.shape)
    cont[~bad] = top[~bad] / bot[~bad]
    return cont