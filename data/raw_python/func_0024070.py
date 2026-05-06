def wrap_fmin_slsqp(fun, guess, opt_kwargs={}):
    """Wrapper for :py:func:`fmin_slsqp` to allow it to be called with :py:func:`minimize`-like syntax.

    This is included to enable the code to run with :py:mod:`scipy` versions
    older than 0.11.0.

    Accepts `opt_kwargs` in the same format as used by
    :py:func:`scipy.optimize.minimize`, with the additional precondition
    that the keyword `method` has already been removed by the calling code.

    Parameters
    ----------
    fun : callable
        The function to minimize.
    guess : sequence
        The initial guess for the parameters.
    opt_kwargs : dict, optional
        Dictionary of extra keywords to pass to
        :py:func:`scipy.optimize.minimize`. Refer to that function's
        docstring for valid options. The keywords 'jac', 'hess' and 'hessp'
        are ignored. Note that if you were planning to use `jac` = True
        (i.e., optimization function returns Jacobian) and have set
        `args` = (True,) to tell :py:meth:`update_hyperparameters` to
        compute and return the Jacobian this may cause unexpected behavior.
        Default is: {}.

    Returns
    -------
    Result : namedtuple
        :py:class:`namedtuple` that mimics the fields of the
        :py:class:`Result` object returned by
        :py:func:`scipy.optimize.minimize`. Has the following fields:

        ======= ======= ===================================================================================
        status  int     Code indicating the exit mode of the optimizer (`imode` from :py:func:`fmin_slsqp`)
        success bool    Boolean indicating whether or not the optimizer thinks a minimum was found.
        fun     float   Value of the optimized function (-1*LL).
        x       ndarray Optimal values of the hyperparameters.
        message str     String describing the exit state (`smode` from :py:func:`fmin_slsqp`)
        nit     int     Number of iterations.
        ======= ======= ===================================================================================

    Raises
    ------
    ValueError
        Invalid constraint type in `constraints`. (See documentation for :py:func:`scipy.optimize.minimize`.)
    """
    opt_kwargs = dict(opt_kwargs)

    opt_kwargs.pop('method', None)

    eqcons = []
    ieqcons = []
    if 'constraints' in opt_kwargs:
        if isinstance(opt_kwargs['constraints'], dict):
            opt_kwargs['constraints'] = [opt_kwargs['constraints'],]
        for con in opt_kwargs.pop('constraints'):
            if con['type'] == 'eq':
                eqcons += [con['fun'],]
            elif con['type'] == 'ineq':
                ieqcons += [con['fun'],]
            else:
                raise ValueError("Invalid constraint type %s!" % (con['type'],))

    if 'jac' in opt_kwargs:
        warnings.warn("Jacobian not supported for default solver SLSQP!",
                      RuntimeWarning)
        opt_kwargs.pop('jac')

    if 'tol' in opt_kwargs:
        opt_kwargs['acc'] = opt_kwargs.pop('tol')

    if 'options' in opt_kwargs:
        opts = opt_kwargs.pop('options')
        opt_kwargs = dict(opt_kwargs.items() + opts.items())

    # Other keywords with less likelihood for causing failures are silently ignored:
    opt_kwargs.pop('hess', None)
    opt_kwargs.pop('hessp', None)
    opt_kwargs.pop('callback', None)

    out, fx, its, imode, smode = scipy.optimize.fmin_slsqp(
        fun,
        guess,
        full_output=True,
        eqcons=eqcons,
        ieqcons=ieqcons,
        **opt_kwargs
    )

    Result = collections.namedtuple('Result',
                                    ['status', 'success', 'fun', 'x', 'message', 'nit'])

    return Result(status=imode,
                  success=(imode == 0),
                  fun=fx,
                  x=out,
                  message=smode,
                  nit=its)