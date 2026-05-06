def forward(A, pobs, pi, T=None, alpha_out=None, dtype=np.float32):
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
        container for the alpha result variables. If None, a new container will be created.
    dtype : type, optional, default = np.float32
        data type of the result.

    Returns
    -------
    logprob : float
        The probability to observe the sequence `ob` with the model given
        by `A`, `B` and `pi`.
    alpha : ndarray((T,N), dtype = float), optional, default = None
        alpha[t,i] is the ith forward coefficient of time t. These can be
        used in many different algorithms related to HMMs.

    """
    # set T
    if T is None:
        T = pobs.shape[0]  # if not set, use the length of pobs as trajectory length
    elif T > pobs.shape[0]:
        raise ValueError('T must be at most the length of pobs.')
    # set N
    N = A.shape[0]
    # initialize output if necessary
    if alpha_out is None:
        alpha_out = np.zeros((T, N), dtype=dtype)
    elif T > alpha_out.shape[0]:
        raise ValueError('alpha_out must at least have length T in order to fit trajectory.')
    # log-likelihood
    logprob = 0.0

    # initial values
    # alpha_i(0) = pi_i * B_i,ob[0]
    np.multiply(pi, pobs[0, :], out=alpha_out[0])
    # scaling factor
    scale = np.sum(alpha_out[0, :])
    # scale
    alpha_out[0, :] /= scale
    logprob += np.log(scale)

    # induction
    for t in range(T-1):
        # alpha_j(t+1) = sum_i alpha_i(t) * A_i,j * B_j,ob(t+1)
        np.multiply(np.dot(alpha_out[t, :], A), pobs[t+1, :], out=alpha_out[t+1])
        # scaling factor
        scale = np.sum(alpha_out[t+1, :])
        # scale
        alpha_out[t+1, :] /= scale
        # update logprob
        logprob += np.log(scale)

    return logprob, alpha_out