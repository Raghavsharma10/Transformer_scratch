def covar_errors(params, data, errs, B, C=None):
    """
    Take a set of parameters that were fit with lmfit, and replace the errors
    with the 1\sigma errors calculated using the covariance matrix.


    Parameters
    ----------
    params : lmfit.Parameters
        Model

    data : 2d-array
        Image data

    errs : 2d-array ?
        Image noise.

    B : 2d-array
        B matrix.

    C : 2d-array
        C matrix. Optional. If supplied then Bmatrix will not be used.

    Returns
    -------
    params : lmfit.Parameters
        Modified model.
    """

    mask = np.where(np.isfinite(data))

    # calculate the proper parameter errors and copy them across.
    if C is not None:
        try:
            J = lmfit_jacobian(params, mask[0], mask[1], errs=errs)
            covar = np.transpose(J).dot(inv(C)).dot(J)
            onesigma = np.sqrt(np.diag(inv(covar)))
        except (np.linalg.linalg.LinAlgError, ValueError) as _:
            C = None

    if C is None:
        try:
            J = lmfit_jacobian(params, mask[0], mask[1], B=B, errs=errs)
            covar = np.transpose(J).dot(J)
            onesigma = np.sqrt(np.diag(inv(covar)))
        except (np.linalg.linalg.LinAlgError, ValueError) as _:
            onesigma = [-2] * len(mask[0])

    for i in range(params['components'].value):
        prefix = "c{0}_".format(i)
        j = 0
        for p in ['amp', 'xo', 'yo', 'sx', 'sy', 'theta']:
            if params[prefix + p].vary:
                params[prefix + p].stderr = onesigma[j]
                j += 1

    return params