def findbeam_azimuthal(data, orig_initial, mask=None, maxiter=100, Ntheta=50,
                       dmin=0, dmax=np.inf, extent=10, callback=None):
    """Find beam center using azimuthal integration

    Inputs:
        data: scattering matrix
        orig_initial: estimated value for x (row) and y (column)
            coordinates of the beam center, starting from 1.
        mask: mask matrix. If None, nothing will be masked. Otherwise it
            should be of the same size as data. Nonzero means non-masked.
        maxiter: maximum number of iterations for scipy.optimize.fmin
        Ntheta: the number of theta points for the azimuthal integration
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
    if mask is None:
        mask = np.ones(data.shape)
    data = data.astype(np.double)

    def targetfunc(orig, data, mask, orig_orig, callback):
        def sinfun(p, x, y):
            return (y - np.sin(x + p[1]) * p[0] - p[2]) / np.sqrt(len(x))
        t, I, a = azimintpix(data, None, orig[
                             0] + orig_orig[0], orig[1] + orig_orig[1], mask.astype('uint8'), Ntheta, dmin, dmax)
        if len(a) > (a > 0).sum():
            raise ValueError('findbeam_azimuthal: non-complete azimuthal average, please consider changing dmin, dmax and/or orig_initial!')
        p = ((I.max() - I.min()) / 2.0, t[I == I.max()][0], I.mean())
        p = scipy.optimize.leastsq(sinfun, p, (t, I))[0]
        # print "findbeam_azimuthal: orig=",orig,"amplitude=",abs(p[0])
        if callback is not None:
            callback()
        return abs(p[0])
    orig1 = scipy.optimize.fmin(targetfunc, np.array([extent, extent]),
                                args=(data, 1 - mask, np.array(orig_initial) - extent,
                                      callback), maxiter=maxiter, disp=0)
    return orig1 + np.array(orig_initial) - extent