def findpeak_asymmetric(x, y, dy=None, curve='Lorentz', return_x=None, init_parameters=None):
    """Find an asymmetric Lorentzian peak.

    Inputs:
        x: numpy array of the abscissa
        y: numpy array of the ordinate
        dy: numpy array of the errors in y (or None if not present)
        curve: string (case insensitive): if starts with "Lorentz",
            a Lorentzian curve will be fitted. If starts with "Gauss",
            a Gaussian will be fitted. Otherwise error.
        return_x: numpy array of the x values at which the best
            fitting peak function should be evaluated and returned
        init_parameters: either None, or a list of [amplitude, center,
            hwhm_left, hwhm_right, baseline]: initial parameters to
            start fitting from.

    Results: center, hwhm_left, hwhm_right, baseline, amplitude [, y_fitted]
        The fitted parameters are returned as floats if dy was None or
        ErrorValue instances if dy was not None.
        y_fitted is only returned if return_x was not None

    Note:
        1) The dataset must contain only the peak.
        2) A positive peak will be fitted
        3) The peak center must be in the given range
    """
    idx = np.logical_and(np.isfinite(x), np.isfinite(y))
    if dy is not None:
        idx = np.logical_and(idx, np.isfinite(dy))
    x=x[idx]
    y=y[idx]
    if dy is not None:
        dy=dy[idx]
    if curve.lower().startswith('loren'):
        lorentzian = True
    elif curve.lower().startswith('gauss'):
        lorentzian = False
    else:
        raise ValueError('Unknown peak type {}'.format(curve))

    def peakfunc(pars, x, lorentzian=True):
        x0, sigma1, sigma2, C, A = pars
        result = np.empty_like(x)
        if lorentzian:
            result[x < x0] = A * sigma1 ** 2 / (sigma1 ** 2 + (x0 - x[x < x0]) ** 2) + C
            result[x >= x0] = A * sigma2 ** 2 / (sigma2 ** 2 + (x0 - x[x >= x0]) ** 2) + C
        else:
            result[x < x0] = A * np.exp(-(x[x < x0] - x0) ** 2 / (2 * sigma1 ** 2))
            result[x >= x0] = A * np.exp(-(x[x >= x0] - x0) ** 2 / (2 * sigma1 ** 2))
        return result

    def fitfunc(pars, x, y, dy, lorentzian=True):
        yfit = peakfunc(pars, x, lorentzian)
        if dy is None:
            return yfit - y
        else:
            return (yfit - y) / dy
    if init_parameters is not None:
        pos, hwhmleft, hwhmright, baseline, amplitude = [float(x) for x in init_parameters]
    else:
        baseline = y.min()
        amplitude = y.max() - baseline
        hwhmleft = hwhmright = (x.max() - x.min()) * 0.5
        pos = x[np.argmax(y)]
    #print([pos,hwhm,hwhm,baseline,amplitude])
    result = scipy.optimize.least_squares(fitfunc, [pos, hwhmleft, hwhmright, baseline, amplitude],
                                          args=(x, y, dy, lorentzian),
                                          bounds=([x.min(), 0, 0, -np.inf, 0],
                                                  [x.max(), np.inf, np.inf, np.inf, np.inf]))
#    print(result.x[0], result.x[1], result.x[2], result.x[3], result.x[4], result.message, result.success)
    if not result.success:
        raise RuntimeError('Error while peak fitting: {}'.format(result.message))
    if dy is None:
        ret = (result.x[0], result.x[1], result.x[2], result.x[3], result.x[4])
    else:
        # noinspection PyTupleAssignmentBalance
        _, s, VT = svd(result.jac, full_matrices=False)
        threshold = np.finfo(float).eps * max(result.jac.shape) * s[0]
        s = s[s > threshold]
        VT = VT[:s.size]
        pcov = np.dot(VT.T / s ** 2, VT)
        ret = tuple([ErrorValue(result.x[i], pcov[i, i] ** 0.5) for i in range(5)])
    if return_x is not None:
        ret = ret + (peakfunc([float(x) for x in ret], return_x, lorentzian),)
    return ret