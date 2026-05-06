def simultaneous_nonlinear_leastsquares(xs, ys, dys, func, params_inits, verbose=False, **kwargs):
    """Do a simultaneous nonlinear least-squares fit and return the fitted
    parameters as instances of ErrorValue.

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
    `parset1, parset2 ...`: tuples of fitted parameters corresponding to curve1,
        curve2, etc. Each tuple contains the values of the fitted parameters
        as instances of ErrorValue, in the same order as they are in
        `params_init`.
    `statdict`: statistics dictionary. This is of the same form as in
        `nlsq_fit`, except that func_value is a sequence of one-dimensional
        np.ndarrays containing the best-fitting function values for each curve.
    """
    p, dp, statdict = simultaneous_nlsq_fit(xs, ys, dys, func, params_inits,
                                            verbose, **kwargs)
    params = [[ErrorValue(p_, dp_) for (p_, dp_) in zip(pcurrent, dpcurrent)]
              for (pcurrent, dpcurrent) in zip(p, dp)]
    return tuple(params + [statdict])