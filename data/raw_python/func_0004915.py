def findbeam_radialpeak(data, orig_initial, mask, rmin, rmax, maxiter=100,
                        drive_by='amplitude', extent=10, callback=None):
    """Find the beam by minimizing the width of a peak in the radial average.

    Inputs:
        data: scattering matrix
        orig_initial: first guess for the origin
        mask: mask matrix. Nonzero is non-masked.
        rmin,rmax: distance from the origin (in pixels) of the peak range.
        drive_by: 'hwhm' to minimize the hwhm of the peak or 'amplitude' to
            maximize the peak amplitude
        extent: approximate distance of the current and the real origin in pixels.
            Too high a value makes the fitting procedure unstable. Too low a value
            does not permit to move away the current origin.
        callback: callback function (expects no arguments)
    Outputs:
        the beam coordinates

    Notes:
        A Gaussian will be fitted.
    """
    orig_initial = np.array(orig_initial)
    mask = 1 - mask.astype(np.uint8)
    data = data.astype(np.double)
    pix = np.arange(rmin * 1.0, rmax * 1.0, 1)
    if drive_by.lower() == 'hwhm':
        def targetfunc(orig, data, mask, orig_orig, callback):
            I = radintpix(
                data, None, orig[0] + orig_orig[0], orig[1] + orig_orig[1], mask, pix)[1]
            hwhm = float(misc.findpeak_single(pix, I)[1])
            # print orig[0] + orig_orig[0], orig[1] + orig_orig[1], p
            if callback is not None:
                callback()
            return abs(hwhm)
    elif drive_by.lower() == 'amplitude':
        def targetfunc(orig, data, mask, orig_orig, callback):
            I = radintpix(
                data, None, orig[0] + orig_orig[0], orig[1] + orig_orig[1], mask, pix)[1]
            fp = misc.findpeak_single(pix, I)
            height = -float(fp[2] + fp[3])
            # print orig[0] + orig_orig[0], orig[1] + orig_orig[1], p
            if callback is not None:
                callback()
            return height
    else:
        raise ValueError('Invalid argument for drive_by %s' % drive_by)
    orig1 = scipy.optimize.fmin(targetfunc, np.array([extent, extent]),
                                args=(
                                    data, mask, orig_initial - extent, callback),
                                maxiter=maxiter, disp=0)
    return np.array(orig_initial) - extent + orig1