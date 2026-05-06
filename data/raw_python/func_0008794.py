def bias_correct(params, data, acf=None):
    """
    Calculate and apply a bias correction to the given fit parameters


    Parameters
    ----------
    params : lmfit.Parameters
        The model parameters. These will be modified.

    data : 2d-array
        The data which was used in the fitting

    acf : 2d-array
        ACF of the data. Default = None.

    Returns
    -------
    None

    See Also
    --------
    :func:`AegeanTools.fitting.RB_bias`
    """
    bias = RB_bias(data, params, acf=acf)
    i = 0
    for p in params:
        if 'theta' in p:
            continue
        if params[p].vary:
            params[p].value -= bias[i]
            i += 1
    return