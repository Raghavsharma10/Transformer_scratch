def jacobian(pars, x, y):
    """
    Analytical calculation of the Jacobian for an elliptical gaussian
    Will work for a model that contains multiple Gaussians, and for which
    some components are not being fit (don't vary).

    Parameters
    ----------
    pars : lmfit.Model
        The model parameters
    x, y : list
        Locations at which the jacobian is being evaluated

    Returns
    -------
    j : 2d array
        The Jacobian.

    See Also
    --------
    :func:`AegeanTools.fitting.emp_jacobian`
    """

    matrix = []

    for i in range(pars['components'].value):
        prefix = "c{0}_".format(i)
        amp = pars[prefix + 'amp'].value
        xo = pars[prefix + 'xo'].value
        yo = pars[prefix + 'yo'].value
        sx = pars[prefix + 'sx'].value
        sy = pars[prefix + 'sy'].value
        theta = pars[prefix + 'theta'].value

        # The derivative with respect to component i doesn't depend on any other components
        # thus the model should not contain the other components
        model = elliptical_gaussian(x, y, amp, xo, yo, sx, sy, theta)

        # precompute for speed
        sint = np.sin(np.radians(theta))
        cost = np.cos(np.radians(theta))
        xxo = x - xo
        yyo = y - yo
        xcos, ycos = xxo * cost, yyo * cost
        xsin, ysin = xxo * sint, yyo * sint

        if pars[prefix + 'amp'].vary:
            dmds = model / amp
            matrix.append(dmds)

        if pars[prefix + 'xo'].vary:
            dmdxo = cost * (xcos + ysin) / sx ** 2 + sint * (xsin - ycos) / sy ** 2
            dmdxo *= model
            matrix.append(dmdxo)

        if pars[prefix + 'yo'].vary:
            dmdyo = sint * (xcos + ysin) / sx ** 2 - cost * (xsin - ycos) / sy ** 2
            dmdyo *= model
            matrix.append(dmdyo)

        if pars[prefix + 'sx'].vary:
            dmdsx = model / sx ** 3 * (xcos + ysin) ** 2
            matrix.append(dmdsx)

        if pars[prefix + 'sy'].vary:
            dmdsy = model / sy ** 3 * (xsin - ycos) ** 2
            matrix.append(dmdsy)

        if pars[prefix + 'theta'].vary:
            dmdtheta = model * (sy ** 2 - sx ** 2) * (xsin - ycos) * (xcos + ysin) / sx ** 2 / sy ** 2
            matrix.append(dmdtheta)

    return np.array(matrix)