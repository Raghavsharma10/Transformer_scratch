def sample_path(alpha, A, pobs, T=None):
    """ Sample the hidden pathway S from the conditional distribution P ( S | Parameters, Observations )

    Parameters
    ----------
    alpha : ndarray((T,N), dtype = float), optional, default = None
        alpha[t,i] is the ith forward coefficient of time t.
    A : ndarray((N,N), dtype = float)
        transition matrix of the hidden states
    pobs : ndarray((T,N), dtype = float)
        pobs[t,i] is the observation probability for observation at time t given hidden state i
    T : int
        number of time steps

    Returns
    -------
    S : numpy.array shape (T)
        maximum likelihood hidden path

    """
    if __impl__ == __IMPL_PYTHON__:
        return ip.sample_path(alpha, A, pobs, T=T, dtype=config.dtype)
    elif __impl__ == __IMPL_C__:
        return ic.sample_path(alpha, A, pobs, T=T, dtype=config.dtype)
    else:
        raise RuntimeError('Nonexisting implementation selected: '+str(__impl__))