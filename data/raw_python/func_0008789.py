def lmfit_jacobian(pars, x, y, errs=None, B=None, emp=False):
    """
    Wrapper around :func:`AegeanTools.fitting.jacobian` and :func:`AegeanTools.fitting.emp_jacobian`
    which gives the output in a format that is required for lmfit.

    Parameters
    ----------
    pars : lmfit.Model
        The model parameters

    x, y : list
        Locations at which the jacobian is being evaluated

    errs : list
        a vector of 1\sigma errors (optional). Default = None

    B : 2d-array
        a B-matrix (optional) see :func:`AegeanTools.fitting.Bmatrix`

    emp : bool
        If true the use the empirical Jacobian, otherwise use the analytical one.
        Default = False.

    Returns
    -------
    j : 2d-array
        A Jacobian.

    See Also
    --------
    :func:`AegeanTools.fitting.Bmatrix`
    :func:`AegeanTools.fitting.jacobian`
    :func:`AegeanTools.fitting.emp_jacobian`

    """
    if emp:
        matrix = emp_jacobian(pars, x, y)
    else:
        # calculate in the normal way
        matrix = jacobian(pars, x, y)
    # now munge this to be as expected for lmfit
    matrix = np.vstack(matrix)

    if errs is not None:
        matrix /= errs
        # matrix = matrix.dot(errs)

    if B is not None:
        matrix = matrix.dot(B)

    matrix = np.transpose(matrix)
    return matrix