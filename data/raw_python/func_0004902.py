def findpeak_single(x, y, dy=None, position=None, hwhm=None, baseline=None, amplitude=None, curve='Lorentz',
                    return_stat=False, signs=(-1, 1), return_x=None):
    """Find a (positive or negative) peak in the dataset.

    Inputs:
        x, y, dy: abscissa, ordinate and the error of the ordinate (can be None)
        position, hwhm, baseline, amplitude: first guesses for the named parameters
        curve: 'Gauss' or 'Lorentz' (default)
        return_stat: return fitting statistics from easylsq.nlsq_fit()
        signs: a tuple, can be (1,), (-1,), (1,-1). Will try these signs for the peak amplitude
        return_x: abscissa on which the fitted function form has to be evaluated

    Outputs:
        peak position, hwhm, baseline, amplitude[, stat][, peakfunction]

        where:
            peak position, hwhm, baseline, amplitude are ErrorValue instances.
            stat is the statistics dictionary, returned only if return_stat is True
            peakfunction is the fitted peak evaluated at return_x if it is not None.

    Notes:
        A Gauss or a Lorentz curve is fitted, depending on the value of 'curve'. The abscissa
        should be sorted, ascending.
    """
    y_orig=y
    if dy is None: dy = np.ones_like(x)
    if curve.upper().startswith('GAUSS'):
        def fitfunc(x_, amplitude_, position_, hwhm_, baseline_):
            return amplitude_ * np.exp(-0.5 * (x_ - position_) ** 2 / hwhm_ ** 2) + baseline_
    elif curve.upper().startswith('LORENTZ'):
        def fitfunc(x_, amplitude_, position_, hwhm_, baseline_):
            return amplitude_ * hwhm_ ** 2 / (hwhm_ ** 2 + (position_ - x_) ** 2) + baseline_
    else:
        raise ValueError('Invalid curve type: {}'.format(curve))
    results=[]
    # we try fitting a positive and a negative peak and return the better fit (where R2 is larger)
    for sign in signs:
        init_params={'position':position,'hwhm':hwhm,'baseline':baseline,'amplitude':amplitude}
        y = y_orig * sign
        if init_params['position'] is None: init_params['position'] = x[y == y.max()][0]
        if init_params['hwhm'] is None: init_params['hwhm'] = 0.5 * (x.max() - x.min())
        if init_params['baseline'] is None: init_params['baseline'] = y.min()
        if init_params['amplitude'] is None: init_params['amplitude'] = y.max() - init_params['baseline']
        results.append(nlsq_fit(x, y, dy, fitfunc, (init_params['amplitude'],
                                                   init_params['position'],
                                                   init_params['hwhm'],
                                                   init_params['baseline']))+(sign,))
    max_R2=max([r[2]['R2'] for r in results])
    p,dp,stat,sign=[r for r in results if r[2]['R2']==max_R2][0]
    retval = [ErrorValue(p[1], dp[1]), ErrorValue(abs(p[2]), dp[2]), sign * ErrorValue(p[3], dp[3]),
              sign * ErrorValue(p[0], dp[0])]
    if return_stat:
        stat['func_value'] = stat['func_value'] * sign
        retval.append(stat)
    if return_x is not None:
        retval.append(sign * fitfunc(return_x, p[0], p[1], p[2], p[3]))
    return tuple(retval)