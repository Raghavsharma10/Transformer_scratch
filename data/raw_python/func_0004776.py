def _find_cont_fitfunc(fluxes, ivars, contmask, deg, ffunc, n_proc=1):
    """ Fit a continuum to a continuum pixels in a segment of spectra

    Functional form can be either sinusoid or chebyshev, with specified degree

    Parameters
    ----------
    fluxes: numpy ndarray of shape (nstars, npixels)
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

    if n_proc == 1:
        for jj in range(nstars):
            flux = fluxes[jj,:]
            ivar = ivars[jj,:]
            pix = np.arange(0, npixels)
            y = flux[contmask]
            x = pix[contmask]
            yivar = ivar[contmask]
            yivar[yivar == 0] = SMALL**2
            if ffunc=="sinusoid":
                p0 = np.ones(deg*2) # one for cos, one for sin
                L = max(x)-min(x)
                pcont_func = _partial_func(_sinusoid, L=L, y=flux)
                popt, pcov = opt.curve_fit(pcont_func, x, y, p0=p0,
                                           sigma=1./np.sqrt(yivar))
            elif ffunc=="chebyshev":
                fit = np.polynomial.chebyshev.Chebyshev.fit(x=x,y=y,w=yivar,deg=deg)
            for element in pix:
                if ffunc=="sinusoid":
                    cont[jj,element] = _sinusoid(element, popt, L=L, y=flux)
                elif ffunc=="chebyshev":
                    cont[jj,element] = fit(element)
    else:
        # start mp.Pool
        pool = mp.Pool(processes=n_proc)
        mp_results = []
        for i in xrange(nstars):
            mp_results.append(pool.apply_async(\
                _find_cont_fitfunc,
                (fluxes[i, :].reshape((1, -1)),
                 ivars[i, :].reshape((1, -1)),
                 contmask[:]),
                {'deg':deg, 'ffunc':ffunc}))
        # close mp.Pool
        pool.close()
        pool.join()

        cont = np.array([mp_results[i].get().flatten() for i in xrange(nstars)])

    return cont