def nonlinear_odr(x, y, dx, dy, func, params_init, **kwargs):
    """Perform a non-linear orthogonal distance regression, return the results as
    ErrorValue() instances.

    Inputs:
        x: one-dimensional numpy array of the independent variable
        y: one-dimensional numpy array of the dependent variable
        dx: absolute error (square root of the variance) of the independent
            variable. Either a one-dimensional numpy array or None. If None,
            weighting is disabled. Non-finite (NaN or inf) elements signify
            that the corresponding element in x is to be treated as fixed by
            ODRPACK.
        dy: absolute error (square root of the variance) of the dependent
            variable. Either a one-dimensional numpy array or None. If None,
            weighting is disabled.
        func: a callable with the signature
            func(x,par1,par2,par3,...)
        params_init: list or tuple of the first estimates of the
            parameters par1, par2, par3 etc. to be fitted

        other optional keyword arguments will be passed to leastsq().

    Outputs: par1, par2, par3, ... , statdict
        par1, par2, par3, ...: fitted values of par1, par2, par3 etc
            as instances of ErrorValue.
        statdict: dictionary of various statistical parameters:
            'DoF': Degrees of freedom
            'Chi2': Chi-squared
            'Chi2_reduced': Reduced Chi-squared
            'num_func_eval': number of function evaluations during fit.
            'func_value': the function evaluated in the best fitting parameters
            'message': status message from leastsq()
            'error_flag': integer status flag from leastsq() ('ier')
            'Covariance': covariance matrix (variances in the diagonal)
            'Correlation_coeffs': Pearson's correlation coefficients (usually
                denoted by 'r') in a matrix. The diagonal is unity.

    Notes:
        for the actual fitting, the module scipy.odr is used.
    """
    odrmodel=odr.Model(lambda pars, x: func(x,*pars))
    if dx is not None:
        # treat non-finite values as fixed
        xfixed=np.isfinite(dx)
    else:
        xfixed=None

    odrdata=odr.RealData(x, y, sx=dx,sy=dy, fix=xfixed)
    odrodr=odr.ODR(odrdata,odrmodel,params_init,ifixb=[not(isinstance(p,FixedParameter)) for p in params_init],
                   **kwargs)
    odroutput=odrodr.run()
    statdict=odroutput.__dict__.copy()
    statdict['Covariance']=odroutput.cov_beta
    statdict['Correlation_coeffs']=odroutput.cov_beta/np.outer(odroutput.sd_beta,odroutput.sd_beta)
    statdict['DoF']=len(x)-len(odroutput.beta)
    statdict['Chi2_reduced']=statdict['res_var']
    statdict['func_value']=statdict['y']
    statdict['Chi2']=statdict['sum_square']
    def convert(p_, dp_, pi):
        if isinstance(pi, FixedParameter):
            return FixedParameter(p_)
        else:
            return ErrorValue(p_, dp_)
    return tuple([convert(p_, dp_, pi) for (p_, dp_, pi) in zip(odroutput.beta, odroutput.sd_beta, params_init)] + [statdict])