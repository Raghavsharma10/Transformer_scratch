def do_lmfit(data, params, B=None, errs=None, dojac=True):
    """
    Fit the model to the data
    data may contain 'flagged' or 'masked' data with the value of np.NaN

    Parameters
    ----------
    data : 2d-array
        Image data

    params : lmfit.Parameters
        Initial model guess.

    B : 2d-array
        B matrix to be used in residual calculations.
        Default = None.

    errs : 1d-array

    dojac : bool
        If true then an analytic jacobian will be passed to the fitting routine.

    Returns
    -------
    result : ?
        lmfit.minimize result.

    params : lmfit.Params
        Fitted model.

    See Also
    --------
    :func:`AegeanTools.fitting.lmfit_jacobian`

    """
    # copy the params so as not to change the initial conditions
    # in case we want to use them elsewhere
    params = copy.deepcopy(params)
    data = np.array(data)
    mask = np.where(np.isfinite(data))

    def residual(params, **kwargs):
        """
        The residual function required by lmfit

        Parameters
        ----------
        params: lmfit.Params
            The parameters of the model being fit

        Returns
        -------
        result : numpy.ndarray
            Model - Data
        """
        f = ntwodgaussian_lmfit(params)  # A function describing the model
        model = f(*mask)  # The actual model
        if B is None:
            return model - data[mask]
        else:
            return (model - data[mask]).dot(B)

    if dojac:
        result = lmfit.minimize(residual, params, kws={'x': mask[0], 'y': mask[1], 'B': B, 'errs': errs}, Dfun=lmfit_jacobian)
    else:
        result = lmfit.minimize(residual, params, kws={'x': mask[0], 'y': mask[1], 'B': B, 'errs': errs})

    # Remake the residual so that it is once again (model - data)
    if B is not None:
        result.residual = result.residual.dot(inv(B))
    return result, params