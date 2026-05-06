def transition_counts(alpha, beta, A, pobs, T=None, out=None):
    """ Sum for all t the probability to transition from state i to state j.

    Parameters
    ----------
    alpha : ndarray((T,N), dtype = float), optional, default = None
        alpha[t,i] is the ith forward coefficient of time t.
    beta : ndarray((T,N), dtype = float), optional, default = None
        beta[t,i] is the ith forward coefficient of time t.
    A : ndarray((N,N), dtype = float)
        transition matrix of the hidden states
    pobs : ndarray((T,N), dtype = float)
        pobs[t,i] is the observation probability for observation at time t given hidden state i
    T : int
        number of time steps
    out : ndarray((N,N), dtype = float), optional, default = None
        containter for the resulting count matrix. If None, a new matrix will be created.

    Returns
    -------
    counts : numpy.array shape (N, N)
         counts[i, j] is the summed probability to transition from i to j in time [0,T)

    See Also
    --------
    forward : calculate forward coefficients `alpha`
    backward : calculate backward coefficients `beta`

    """
    if __impl__ == __IMPL_PYTHON__:
        return ip.transition_counts(alpha, beta, A, pobs, T=T, out=out, dtype=config.dtype)
    elif __impl__ == __IMPL_C__:
        return ic.transition_counts(alpha, beta, A, pobs, T=T, out=out, dtype=config.dtype)
    else:
        raise RuntimeError('Nonexisting implementation selected: '+str(__impl__))