def _find_cont_fitfunc_regions(fluxes, ivars, contmask, deg, ranges, ffunc,
                               n_proc=1):
    """ Run fit_cont, dealing with spectrum in regions or chunks

    This is useful if a spectrum has gaps.

    Parameters
    ----------
    fluxes: ndarray of shape (nstars, npixels)
        training set or test set pixel intensities

    ivars: numpy ndarray of shape (nstars, npixels)
        inverse variances, parallel to fluxes

    contmask: numpy ndarray of length (npixels)
        boolean pixel mask, True indicates that pixel is continuum 

    deg: int
        degree of fitting function

    ffunc: str
        type of fitting function, chebyshev or sinusoid

    Returns
    -------
    cont: numpy ndarray of shape (nstars, npixels)
        the continuum, parallel to fluxes
    """
    nstars = fluxes.shape[0]
    npixels = fluxes.shape[1]
    cont = np.zeros(fluxes.shape)
    for chunk in ranges:
        start = chunk[0]
        stop = chunk[1]
        if ffunc=="chebyshev":
            output = _find_cont_fitfunc(fluxes[:,start:stop],
                                        ivars[:,start:stop],
                                        contmask[start:stop],
                                        deg=deg, ffunc="chebyshev",
                                        n_proc=n_proc)
        elif ffunc=="sinusoid":
            output = _find_cont_fitfunc(fluxes[:,start:stop],
                                        ivars[:,start:stop],
                                        contmask[start:stop],
                                        deg=deg, ffunc="sinusoid",
                                        n_proc=n_proc)
        cont[:, start:stop] = output

    return cont