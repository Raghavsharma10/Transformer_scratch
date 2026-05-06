def emp_hessian(pars, x, y):
    """
    Calculate the hessian matrix empirically.
    Create a hessian matrix corresponding to the source model 'pars'
    Only parameters that vary will contribute to the hessian.
    Thus there will be a total of nvar x nvar entries, each of which is a
    len(x) x len(y) array.

    Parameters
    ----------
    pars : lmfit.Parameters
        The model
    x, y : list
        locations at which to evaluate the Hessian

    Returns
    -------
    h : np.array
        Hessian. Shape will be (nvar, nvar, len(x), len(y))

    Notes
    -----
    Uses :func:`AegeanTools.fitting.emp_jacobian` to calculate the first order derivatives.

    See Also
    --------
    :func:`AegeanTools.fitting.hessian`
    """
    eps = 1e-5
    matrix = []
    for i in range(pars['components'].value):
        model = emp_jacobian(pars, x, y)
        prefix = "c{0}_".format(i)
        for p in ['amp', 'xo', 'yo', 'sx', 'sy', 'theta']:
            if pars[prefix+p].vary:
                pars[prefix+p].value += eps
                dm2didj = emp_jacobian(pars, x, y) - model
                matrix.append(dm2didj/eps)
                pars[prefix+p].value -= eps
    matrix = np.array(matrix)
    return matrix