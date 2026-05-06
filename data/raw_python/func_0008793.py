def RB_bias(data, pars, ita=None, acf=None):
    """
    Calculate the expected bias on each of the parameters in the model pars.
    Only parameters that are allowed to vary will have a bias.
    Calculation follows the description of Refrieger & Brown 1998 (cite).


    Parameters
    ----------
    data : 2d-array
        data that was fit

    pars : lmfit.Parameters
        The model

    ita : 2d-array
        The ita matrix (optional).

    acf : 2d-array
        The acf for the data.

    Returns
    -------
    bias : array
        The bias on each of the parameters
    """
    log.info("data {0}".format(data.shape))
    nparams = np.sum([pars[k].vary for k in pars.keys() if k != 'components'])
    # masked pixels
    xm, ym = np.where(np.isfinite(data))
    # all pixels
    x, y = np.indices(data.shape)
    # Create the jacobian as an AxN array accounting for the masked pixels
    j = np.array(np.vsplit(lmfit_jacobian(pars, xm, ym).T, nparams)).reshape(nparams, -1)

    h = hessian(pars, x, y)
    # mask the hessian to be AxAxN array
    h = h[:, :, xm, ym]
    Hij = np.einsum('ik,jk', j, j)
    Dij = np.linalg.inv(Hij)
    Bijk = np.einsum('ip,jkp', j, h)
    Eilkm = np.einsum('il,km', Dij, Dij)

    Cimn_1 =    -1 * np.einsum('krj,ir,km,jn', Bijk, Dij, Dij, Dij)
    Cimn_2 = -1./2 * np.einsum('rkj,ir,km,jn', Bijk, Dij, Dij, Dij)
    Cimn = Cimn_1 + Cimn_2

    if ita is None:
        # N is the noise (data-model)
        N = data - ntwodgaussian_lmfit(pars)(x, y)
        if acf is None:
            acf = nan_acf(N)
        ita = make_ita(N, acf=acf)
        log.info('acf.shape {0}'.format(acf.shape))
        log.info('acf[0] {0}'.format(acf[0]))
        log.info('ita.shape {0}'.format(ita.shape))
        log.info('ita[0] {0}'.format(ita[0]))

    # Included for completeness but not required

    # now mask/ravel the noise
    # N = N[np.isfinite(N)].ravel()
    # Pi = np.einsum('ip,p', j, N)
    # Qij = np.einsum('ijp,p', h, N)

    Vij = np.einsum('ip,jq,pq', j, j, ita)
    Uijk = np.einsum('ip,jkq,pq', j, h, ita)

    bias_1 = np.einsum('imn, mn', Cimn, Vij)
    bias_2 = np.einsum('ilkm, mlk', Eilkm, Uijk)
    bias = bias_1 + bias_2
    log.info('bias {0}'.format(bias))
    return bias