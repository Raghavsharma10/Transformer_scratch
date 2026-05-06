def findbeam_powerlaw(data, orig_initial, mask, rmin, rmax, maxiter=100,
                      drive_by='R2', extent=10, callback=None):
    """Find the beam by minimizing the width of a Gaussian centered at the
    origin (i.e. maximizing the radius of gyration in a Guinier scattering).

    Inputs:
        data: scattering matrix
        orig_initial: first guess for the origin
        mask: mask matrix. Nonzero is non-masked.
        rmin,rmax: distance from the origin (in pixels) of the fitting range
        drive_by: 'R2' or 'Chi2'
        extent: approximate distance of the current and the real origin in pixels.
            Too high a value makes the fitting procedure unstable. Too low a value
            does not permit to move away the current origin.
        callback: callback function (expects no arguments)
    Outputs:
        the beam coordinates

    Notes:
        A power-law will be fitted
    """
    orig_initial = np.array(orig_initial)
    mask = 1 - mask.astype(np.uint8)
    data = data.astype(np.double)
    pix = np.arange(rmin * 1.0, rmax * 1.0, 1)

    def targetfunc(orig, data, mask, orig_orig, callback):
        I, E = radintpix(
            data, None, orig[0] + orig_orig[0], orig[1] + orig_orig[1], mask, pix)[1:3]
        p, dp, stat = misc.easylsq.nlsq_fit(
            pix, I, E, lambda q, A, alpha: A * q ** alpha, [1.0, -3.0])
        if callback is not None:
            callback()
        #        print(orig, orig_orig, orig + orig_orig, stat[drive_by])
        if drive_by == 'R2':
            return 1 - stat['R2']
        elif drive_by.startswith('Chi2'):
            return stat[drive_by]
    orig1 = scipy.optimize.fmin(targetfunc, np.array([extent, extent]),
                                args=(
                                    data, mask, orig_initial - extent, callback),
                                maxiter=maxiter, disp=False)
    return np.array(orig_initial) - extent + orig1