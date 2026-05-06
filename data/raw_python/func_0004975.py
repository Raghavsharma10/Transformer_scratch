def simultaneous_nlsq_fit(xs, ys, dys, func, params_inits, verbose=False,
                             **kwargs):
    """Do a simultaneous nonlinear least-squares fit

    Input:
    ------
    `xs`: tuple of abscissa vectors (1d numpy ndarrays)
    `ys`: tuple of ordinate vectors (1d numpy ndarrays)
    `dys`: tuple of the errors of ordinate vectors (1d numpy ndarrays or Nones)
    `func`: fitting function (the same for all the datasets)
    `params_init`: tuples of *lists* or *tuples* (not numpy ndarrays!) of the
        initial values of the parameters to be fitted. The special value `None`
        signifies that the corresponding parameter is the same as in the
        previous dataset. Of course, none of the parameters of the first dataset
        can be None.
    `verbose`: if various messages useful for debugging should be printed on
        stdout.

    additional keyword arguments get forwarded to nlsq_fit()

    Output:
    -------
    `p`: tuple of a list of fitted parameters
    `dp`: tuple of a list of errors of the fitted parameters
    `statdict`: statistics dictionary. This is of the same form as in
        `nlsq_fit` except that func_value is a sequence of one-dimensional
        np.ndarrays containing the best-fitting function values for each curve.
    """
    if not isinstance(xs, collections.Sequence) or \
        not isinstance(ys, collections.Sequence) or \
        not isinstance(dys, collections.Sequence) or \
        not isinstance(params_inits, collections.Sequence):
        raise ValueError('Parameters `xs`, `ys`, `dys` and `params_inits` should be tuples or lists.')
    Ndata = len(xs)
    if len(ys) != Ndata or len(dys) != Ndata or len(params_inits) != Ndata:
        raise ValueError('Parameters `xs`, `ys`, `dys` and `params_inits` should have the same length.')

    if not all([isinstance(x, collections.Sequence) for x in params_inits]):
        raise ValueError('Elements of `params_inits` should be tuples or Python lists.')
    Ns = set([len(x) for x in params_inits])
    if len(Ns) != 1:
        raise ValueError('Elements of `params_inits` should have the same length.')
    Npar = Ns.pop()
    for i in range(Ndata):
        if dys[i] is None:
            dys[i] = np.ones(len(xs[i]), np.double) * np.nan
    # concatenate the x, y and dy vectors
    xcat = np.concatenate(xs)
    ycat = np.concatenate(ys)
    dycat = np.concatenate(dys)
    # find the start and end indices for each dataset in the concatenated datasets.
    lens = [len(x) for x in xs]
    starts = [int(sum(lens[:i])) for i in range(len(lens))]
    ends = [int(sum(lens[:i + 1])) for i in range(len(lens))]

    # flatten the initial parameter list. A single list is needed, where the
    # constrained parameters occur only once. Of course, we have to do some
    # bookkeeping to be able to find the needed parameters for each sub-range
    # later during the fit.
    paramcat = []  # this will be the concatenated list of parameters
    param_indices = []  # this will have the same structure as params_inits (i.e.
        # a tuple of tuples of ints). Each tuple corresponds to a dataset.
        # Each integer number in each tuple holds
        # the index of the corresponding fit parameter in the 
        # concatenated parameter list.
    for j in range(Ndata):  # for each dataset
        param_indices.append([])
        jorig = j
        for i in range(Npar):
            j = jorig
            while params_inits[j][i] is None and (j >= 0):
                j = j - 1
            if j < 0:
                raise ValueError('None of the parameters in the very first dataset should be `None`.')
            if jorig == j:  # not constrained parameter
                paramcat.append(params_inits[j][i])
                param_indices[jorig].append(len(paramcat) - 1)
            else:
                param_indices[jorig].append(param_indices[j][i])

    if verbose:
        print("Number of datasets for simultaneous fitting:", Ndata)
        print("Total number of data points:", len(xcat))
        print("Number of parameters in each dataset:", Npar)
        print("Total number of parameters:", Ndata * Npar)
        print("Number of independent parameters:", len(paramcat))
    # the flattened function
    def func_flat(x, *params):
        y = []
        for j in range(Ndata):
            if verbose > 1:
                print("Simultaneous fitting: evaluating function for dataset #", j, "/", Ndata)
            pars = [params[i] for i in param_indices[j]]
            y.append(func(x[starts[j]:ends[j]], *pars))
        return np.concatenate(tuple(y))

    # Now we reduced the problem to a single least-squares fit. Carry it out and
    # interpret the results.
    pflat, dpflat, statdictflat = nlsq_fit(xcat, ycat, dycat, func_flat, paramcat, verbose, **kwargs)
    for n in ['func_value', 'R2', 'Chi2', 'Chi2_reduced', 'DoF', 'Covariance', 'Correlation_coeffs']:
        statdictflat[n + '_global'] = statdictflat[n]
        statdictflat[n] = []
    p = []
    dp = []
    for j in range(Ndata):  # unpack the results
        p.append([pflat[i] for i in param_indices[j]])
        dp.append([dpflat[i] for i in param_indices[j]])
        statdictflat['func_value'].append(statdictflat['func_value_global'][starts[j]:ends[j]])
        if np.isfinite(dys[j]).all():
            statdictflat['Chi2'].append((((statdictflat['func_value'][-1] - ys[j]) / dys[j]) ** 2).sum())
            sstot = np.sum((ys[j] - np.mean(ys[j])) ** 2 / dys[j] ** 2)
        else:
            statdictflat['Chi2'].append(((statdictflat['func_value'][-1] - ys[j]) ** 2).sum())
            sstot = np.sum((ys[j] - np.mean(ys[j])) ** 2)
        sserr = statdictflat['Chi2'][-1]
        statdictflat['R2'].append(1 - sserr / sstot)
        statdictflat['DoF'].append(len(xs[j] - len(p[-1])))
        statdictflat['Covariance'].append(slice_covarmatrix(statdictflat['Covariance_global'], param_indices[j]))
        statdictflat['Correlation_coeffs'].append(slice_covarmatrix(statdictflat['Correlation_coeffs_global'], param_indices[j]))
        statdictflat['Chi2_reduced'].append(statdictflat['Chi2'][-1] / statdictflat['DoF'][-1])
    return p, dp, statdictflat