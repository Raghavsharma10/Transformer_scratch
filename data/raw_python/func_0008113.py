def forward(A, pobs, pi, T=None, alpha_out=None):
    """Compute P( obs | A, B, pi ) and all forward coefficients.

    Parameters
    ----------
    A : ndarray((N,N), dtype = float)
        transition matrix of the hidden states
    pobs : ndarray((T,N), dtype = float)
        pobs[t,i] is the observation probability for observation at time t given hidden state i
    pi : ndarray((N), dtype = float)
        initial distribution of hidden states
    T : int, optional, default = None
        trajectory length. If not given, T = pobs.shape[0] will be used.
    alpha_out : ndarray((T,N), dtype = float), optional, default = None
        containter for the alpha result variables. If None, a new container will be created.

    Returns
    -------
    logprob : float
        The probability to observe the sequence `ob` with the model given
        by `A`, `B` and `pi`.
    alpha : ndarray((T,N), dtype = float), optional, default = None
        alpha[t,i] is the ith forward coefficient of time t. These can be
        used in many different algorithms related to HMMs.

    """
    if __impl__ == __IMPL_PYTHON__:
        return ip.forward(A, pobs, pi, T=T, alpha_out=alpha_out, dtype=config.dtype)
    elif __impl__ == __IMPL_C__:
        return ic.forward(A, pobs, pi, T=T, alpha_out=alpha_out, dtype=config.dtype)
    else:
        raise RuntimeError('Nonexisting implementation selected: '+str(__impl__))