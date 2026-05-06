def viterbi(A, pobs, pi, dtype=np.float32):
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
    T, N = pobs.shape[0], pobs.shape[1]
    # temporary viterbi state
    psi = np.zeros((T, N), dtype=int)
    # initialize
    v = pi * pobs[0, :]
    # rescale
    v /= v.sum()
    psi[0] = 0.0
    # iterate
    for t in range(1, T):
        vA = np.dot(np.diag(v), A)
        # propagate v
        v = pobs[t, :] * np.max(vA, axis=0)
        # rescale
        v /= v.sum()
        psi[t] = np.argmax(vA, axis=0)
    # iterate
    q = np.zeros(T, dtype=int)
    q[T-1] = np.argmax(v)
    for t in range(T-2, -1, -1):
        q[t] = psi[t+1, q[t+1]]
    # done
    return q