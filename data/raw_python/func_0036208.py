def fit(ts, fs=[], all_params=[], fit_vars=None,
        alg='leastsq', make_bounded=True):
    """
    Use a minimization algorithm to fit a AstonSeries with
    analytical functions.
    """
    if fit_vars is None:
        fit_vars = [f._peakargs for f in fs]
    initc = [min(ts.values)]
    for f, peak_params, to_fit in zip(fs, all_params, fit_vars):
        if 'v' in to_fit:
            to_fit.remove('v')

        if make_bounded and hasattr(f, '_pbounds'):
            new_v = _to_unbnd_p({i: peak_params[i] for i in to_fit},
                                f._pbounds)
            initc += [new_v[i] for i in to_fit]
        else:
            initc += [peak_params[i] for i in to_fit]

    def errfunc_lsq(fit_params, t, y, all_params):
        # first value in fit_params is baseline
        # fit_y = np.ones(len(t)) * fit_params[0]
        fit_y = np.zeros(len(t))
        param_i = 1
        for f, peak_params, to_fit in zip(fs, all_params, fit_vars):
            for k in to_fit:
                peak_params[k] = fit_params[param_i]
                param_i += 1
            if make_bounded and hasattr(f, '_pbounds'):
                fit_y += f(t, **_to_bound_p(peak_params, f._pbounds))
            else:
                fit_y += f(t, **peak_params)
        return fit_y - y

    def errfunc(p, t, y, all_params):
        return np.sum(errfunc_lsq(p, t, y, all_params) ** 2)

    if alg == 'simplex':
        fit_p, _ = fmin(errfunc, initc, args=(ts.index, ts.values,
                                              peak_params))
#    elif alg == 'anneal':
#        fit_p, _ = anneal(errfunc, initc, args=(ts.index, ts.values,
#                                                peak_params))
    elif alg == 'lbfgsb':
        # TODO: use bounds param
        fitp, _ = fmin_l_bfgs_b(errfunc, fit_p,
                                args=(ts.index, ts.values, peak_params),
                                approx_grad=True)
    elif alg == 'leastsq':
        fit_p, _ = leastsq(errfunc_lsq, initc, args=(ts.index, ts.values,
                                                     all_params))
    # else:
    #     r = minimize(errfunc, initc, \
    #                  args=(ts.index, ts.values, all_params), \
    #                  jac=False, gtol=1e-2)
    #     #if not r['success']:
    #     #    print('Fail:' + str(f))
    #     #    print(r)
    #     #if np.nan in r['x']:  # not r['success']?
    #     #    fit_p = initc
    #     #else:
    #     #    fit_p = r['x']

    fit_pl = fit_p.tolist()
    v = fit_pl.pop(0)  # noqa
    fitted_params = []
    for f, to_fit in zip(fs, fit_vars):
        fit_p_dict = {v: fit_pl.pop(0) for v in to_fit}
        # fit_p_dict['v'] = v
        if make_bounded and hasattr(f, '_pbounds'):
            fitted_params.append(_to_bound_p(fit_p_dict, f._pbounds))
        else:
            fitted_params.append(fit_p_dict)

    # calculate r^2 of the fit
    ss_err = errfunc(fit_p, ts.index, ts.values, fitted_params)
    ss_tot = np.sum((ts.values - np.mean(ts.values)) ** 2)
    r2 = 1 - ss_err / ss_tot
    res = {'r^2': r2}

    return fitted_params, res