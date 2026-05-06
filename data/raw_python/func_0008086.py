def sample_path(alpha, A, pobs, T=None, dtype=np.float32):
    """
    alpha : ndarray((T,N), dtype = float), optional, default = None
        alpha[t,i] is the ith forward coefficient of time t.
    beta : ndarray((T,N), dtype = float), optional, default = None
        beta[t,i] is the ith forward coefficient of time t.
    A : ndarray((N,N), dtype = float)
        transition matrix of the hidden states
    pobs : ndarray((T,N), dtype = float)
        pobs[t,i] is the observation probability for observation at time t given hidden state i
    """
    N = pobs.shape[1]
    # set T
    if T is None:
        T = pobs.shape[0]  # if not set, use the length of pobs as trajectory length
    elif T > pobs.shape[0] or T > alpha.shape[0]:
        raise ValueError('T must be at most the length of pobs and alpha.')

    # initialize path
    S = np.zeros(T, dtype=int)

    # Sample final state.
    psel = alpha[T-1, :]
    psel /= psel.sum()  # make sure it's normalized
    # Draw from this distribution.
    S[T-1] = np.random.choice(range(N), size=1, p=psel)

    # Work backwards from T-2 to 0.
    for t in range(T-2, -1, -1):
        # Compute P(s_t = i | s_{t+1}..s_T).
        psel = alpha[t, :] * A[:, S[t+1]]
        psel /= psel.sum()  # make sure it's normalized
        # Draw from this distribution.
        S[t] = np.random.choice(range(N), size=1, p=psel)

    return S