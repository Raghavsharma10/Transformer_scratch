def nonlinear_leastsquares(x: np.ndarray, y: np.ndarray, dy: np.ndarray, func: Callable, params_init: np.ndarray,
                           verbose: bool = False, **kwargs):
    """Perform a non-linear least squares fit, return the results as
    ErrorValue() instances.

    Inputs:
        x: one-dimensional numpy array of the independent variable
        y: one-dimensional numpy array of the dependent variable
        dy: absolute error (square root of the variance) of the dependent
            variable. Either a one-dimensional numpy array or None. In the array
            case, if any of its elements is NaN, the whole array is treated as
            NaN (= no weighting)
        func: a callable with the signature
            func(x,par1,par2,par3,...)
        params_init: list or tuple of the first estimates of the
            parameters par1, par2, par3 etc. to be fitted
        `verbose`: if various messages useful for debugging should be printed on
            stdout.

        other optional keyword arguments will be passed to leastsq().

    Outputs: par1, par2, par3, ... , statdict
        par1, par2, par3, ...: fitted values of par1, par2, par3 etc
            as instances of ErrorValue.
        statdict: dictionary of various statistical parameters:
            'DoF': Degrees of freedom
            'Chi2': Chi-squared
            'Chi2_reduced': Reduced Chi-squared
            'R2': Coefficient of determination
            'num_func_eval': number of function evaluations during fit.
            'func_value': the function evaluated in the best fitting parameters
            'message': status message from leastsq()
            'error_flag': integer status flag from leastsq() ('ier')
            'Covariance': covariance matrix (variances in the diagonal)
            'Correlation_coeffs': Pearson's correlation coefficients (usually
                denoted by 'r') in a matrix. The diagonal is unity.

    Notes:
        for the actual fitting, nlsq_fit() is used, which in turn delegates the
            job to scipy.optimize.leastsq().
    """
    newfunc, newparinit = hide_fixedparams(func, params_init)
    p, dp, statdict = nlsq_fit(x, y, dy, newfunc, newparinit, verbose, **kwargs)
    p, statdict['Covariance'] = resubstitute_fixedparams(p, params_init, statdict['Covariance'])
    dp, statdict['Correlation_coeffs'] = resubstitute_fixedparams(dp, [type(p_)(0) for p_ in params_init], statdict['Correlation_coeffs'])
    def convert(p_, dp_):
        if isinstance(p_, FixedParameter) or isinstance(dp_, FixedParameter):
            return p_
        else:
            return ErrorValue(p_, dp_)
    return tuple([convert(p_, dp_) for (p_, dp_) in zip(p, dp)] + [statdict])