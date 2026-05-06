def findbeam_azimuthal_fold(data, orig_initial, mask=None, maxiter=100,
                            Ntheta=50, dmin=0, dmax=np.inf, extent=10, callback=None):
    """Find beam center using azimuthal integration and folding

    Inputs:
        data: scattering matrix
        orig_initial: estimated value for x (row) and y (column)
            coordinates of the beam center, starting from 1.
        mask: mask matrix. If None, nothing will be masked. Otherwise it
            should be of the same size as data. Nonzero means non-masked.
        maxiter: maximum number of iterations for scipy.optimize.fmin
        Ntheta: the number of theta points for the azimuthal integration.
            Should be even!
        dmin: pixels nearer to the origin than this will be excluded from
            the azimuthal integration
        dmax: pixels farther from the origin than this will be excluded from
            the azimuthal integration
        extent: approximate distance of the current and the real origin in pixels.
            Too high a value makes the fitting procedure unstable. Too low a value
            does not permit to move away the current origin.
        callback: callback function (expects no arguments)
    Output:
        a vector of length 2 with the x and y coordinates of the origin,
            starting from 1
    """
    if Ntheta % 2:
        raise ValueError('Ntheta should be even!')
    if mask is None:
        mask = np.ones_like(data).astype(np.uint8)
    data = data.astype(np.double)
    # the function to minimize is the sum of squared difference of two halves of
    # the azimuthal integral.

    def targetfunc(orig, data, mask, orig_orig, callback):
        I = azimintpix(data, None, orig[
                       0] + orig_orig[0], orig[1] + orig_orig[1], mask, Ntheta, dmin, dmax)[1]
        if callback is not None:
            callback()
        return np.sum((I[:Ntheta / 2] - I[Ntheta / 2:]) ** 2) / Ntheta
    orig1 = scipy.optimize.fmin(targetfunc, np.array([extent, extent]),
                                args=(data, 1 - mask, np.array(orig_initial) - extent, callback), maxiter=maxiter, disp=0)
    return orig1 + np.array(orig_initial) - extent