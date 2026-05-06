def SysRem(time, flux, err, ncbv=5, niter=50, sv_win=999,
           sv_order=3, **kwargs):
    '''
    Applies :py:obj:`SysRem` to a given set of light curves.

    :param array_like time: The time array for all of the light curves
    :param array_like flux: A 2D array of the fluxes for each of the light \
           curves, shape `(nfluxes, ntime)`
    :param array_like err: A 2D array of the flux errors for each of the \
           light curves, shape `(nfluxes, ntime)`
    :param int ncbv: The number of signals to recover. Default 5
    :param int niter: The number of :py:obj:`SysRem` iterations to perform. \
           Default 50
    :param int sv_win: The Savitsky-Golay filter window size. Default 999
    :param int sv_order: The Savitsky-Golay filter order. Default 3

    '''

    nflx, tlen = flux.shape

    # Get normalized fluxes
    med = np.nanmedian(flux, axis=1).reshape(-1, 1)
    y = flux - med

    # Compute the inverse of the variances
    invvar = 1. / err ** 2

    # The CBVs for this set of fluxes
    cbvs = np.zeros((ncbv, tlen))

    # Recover `ncbv` components
    for n in range(ncbv):

        # Initialize the weights and regressors
        c = np.zeros(nflx)
        a = np.ones(tlen)
        f = y * invvar

        # Perform `niter` iterations
        for i in range(niter):

            # Compute the `c` vector (the weights)
            c = np.dot(f, a) / np.dot(invvar, a ** 2)

            # Compute the `a` vector (the regressors)
            a = np.dot(c, f) / np.dot(c ** 2, invvar)

        # Remove this component from all light curves
        y -= np.outer(c, a)

        # Save this regressor after smoothing it a bit
        if sv_win >= len(a):
            sv_win = len(a) - 1
            if sv_win % 2 == 0:
                sv_win -= 1
        cbvs[n] = savgol_filter(a - np.nanmedian(a), sv_win, sv_order)

    return cbvs