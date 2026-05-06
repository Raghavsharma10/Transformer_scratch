def viterbi(A, pobs, pi):
    """ Estimate the hidden pathway of maximum likelihood using the Viterbi algorithm.

    Parameters
    ----------
    A : ndarray((N,N), dtype = float)
        transition matrix of the hidden states
    pobs : ndarray((T,N), dtype = float)
        pobs[t,i] is the observation probability for observation at time t given hidden state i
    pi : ndarray((N), dtype = float)
        initial distribution of hidden states

    Returns
    -------
    q : numpy.array shape (T)
        maximum likelihood hidden path

    """
    if __impl__ == __IMPL_PYTHON__:
        return ip.viterbi(A, pobs, pi, dtype=config.dtype)
    elif __impl__ == __IMPL_C__:
        return ic.viterbi(A, pobs, pi, dtype=config.dtype)
    else:
        raise RuntimeError('Nonexisting implementation selected: '+str(__impl__))