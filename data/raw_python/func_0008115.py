def state_probabilities(alpha, beta, T=None, gamma_out=None):
    """ Calculate the (T,N)-probabilty matrix for being in state i at time t.

    Parameters
    ----------
    alpha : ndarray((T,N), dtype = float), optional, default = None
        alpha[t,i] is the ith forward coefficient of time t.
    beta : ndarray((T,N), dtype = float), optional, default = None
        beta[t,i] is the ith forward coefficient of time t.
    T : int, optional, default = None
        trajectory length. If not given, gamma_out.shape[0] will be used. If
        gamma_out is neither given, T = alpha.shape[0] will be used.
    gamma_out : ndarray((T,N), dtype = float), optional, default = None
        containter for the gamma result variables. If None, a new container will be created.

    Returns
    -------
    gamma : ndarray((T,N), dtype = float), optional, default = None
        gamma[t,i] is the probabilty at time t to be in state i !


    See Also
    --------
    forward : to calculate `alpha`
    backward : to calculate `beta`

    """
    # get summation helper - we use matrix multiplication with 1's because it's faster than the np.sum function (yes!)
    global ones_size
    if ones_size != alpha.shape[1]:
        global ones
        ones = np.ones(alpha.shape[1])[:, None]
        ones_size = alpha.shape[1]
    #
    if alpha.shape[0] != beta.shape[0]:
        raise ValueError('Inconsistent sizes of alpha and beta.')
    # determine T to use
    if T is None:
        if gamma_out is None:
            T = alpha.shape[0]
        else:
            T = gamma_out.shape[0]
    # compute
    if gamma_out is None:
        gamma_out = alpha * beta
        if T < gamma_out.shape[0]:
            gamma_out = gamma_out[:T]
    else:
        if gamma_out.shape[0] < alpha.shape[0]:
            np.multiply(alpha[:T], beta[:T], gamma_out)
        else:
            np.multiply(alpha, beta, gamma_out)
    # normalize
    np.divide(gamma_out, np.dot(gamma_out, ones), out=gamma_out)
    # done
    return gamma_out