def findbeam_slices(data, orig_initial, mask=None, maxiter=0, epsfcn=0.001,
                    dmin=0, dmax=np.inf, sector_width=np.pi / 9.0, extent=10, callback=None):
    """Find beam center with the "slices" method

    Inputs:
        data: scattering matrix
        orig_initial: estimated value for x (row) and y (column)
            coordinates of the beam center, starting from 1.
        mask: mask matrix. If None, nothing will be masked. Otherwise it
            should be of the same size as data. Nonzero means non-masked.
        maxiter: maximum number of iterations for scipy.optimize.leastsq
        epsfcn: input for scipy.optimize.leastsq
        dmin: disregard pixels nearer to the origin than this
        dmax: disregard pixels farther from the origin than this
        sector_width: width of sectors in radians
        extent: approximate distance of the current and the real origin in pixels.
            Too high a value makes the fitting procedure unstable. Too low a value
            does not permit to move away the current origin.
        callback: callback function (expects no arguments)

    Output:
        a vector of length 2 with the x (row) and y (column) coordinates
         of the origin.
    """
    if mask is None:
        mask = np.ones(data.shape)
    data = data.astype(np.double)

    def targetfunc(orig, data, mask, orig_orig, callback):
        # integrate four sectors
        I = [None] * 4
        p, Ints, A = radint_nsector(data, None, -1, -1, -1, orig[0] + orig_orig[0], orig[1] + orig_orig[1], mask=mask,
                                    phi0=np.pi / 4 - 0.5 * sector_width, dphi=sector_width,
                                    Nsector=4)
        minpix = max(max(p.min(0).tolist()), dmin)
        maxpix = min(min(p.max(0).tolist()), dmax)
        if (maxpix < minpix):
            raise ValueError('The four slices do not overlap! Please give a\
 better approximation for the origin or use another centering method.')
        for i in range(4):
            I[i] = Ints[:, i][(p[:, i] >= minpix) & (p[:, i] <= maxpix)]
        ret = ((I[0] - I[2]) ** 2 + (I[1] - I[3]) ** 2) / (maxpix - minpix)
        if callback is not None:
            callback()
        return ret
    orig = scipy.optimize.leastsq(targetfunc, np.array([extent, extent]),
                                  args=(data, 1 - mask.astype(np.uint8),
                                        np.array(orig_initial) - extent, callback),
                                  maxfev=maxiter, epsfcn=0.01)
    return orig[0] + np.array(orig_initial) - extent