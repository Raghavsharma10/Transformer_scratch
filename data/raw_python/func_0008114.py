def backward(A, pobs, T=None, beta_out=None):
    """Compute all backward coefficients. With scaling!

    Parameters
    ----------
    A : ndarray((N,N), dtype = float)
        transition matrix of the hidden states
    pobs : ndarray((T,N), dtype = float)
        pobs[t,i] is the observation probability for observation at time t given hidden state i
    T : int, optional, default = None
        trajectory length. If not given, T = pobs.shape[0] will be used.
    beta_out : ndarray((T,N), dtype = float), optional, default = None
        containter for the beta result variables. If None, a new container will be created.

    Returns
    -------
    beta : ndarray((T,N), dtype = float), optional, default = None
        beta[t,i] is the ith backward coefficient of time t. These can be
        used in many different algorithms related to HMMs.

    """
    if __impl__ == __IMPL_PYTHON__:
        return ip.backward(A, pobs, T=T, beta_out=beta_out, dtype=config.dtype)
    elif __impl__ == __IMPL_C__:
        return ic.backward(A, pobs, T=T, beta_out=beta_out, dtype=config.dtype)
    else:
        raise RuntimeError('Nonexisting implementation selected: '+str(__impl__))