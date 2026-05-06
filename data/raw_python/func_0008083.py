def backward(A, pobs, T=None, beta_out=None, dtype=np.float32):
    """Compute all backward coefficients. With scaling!

    Parameters
    ----------
    A : ndarray((N,N), dtype = float)
        transition matrix of the hidden states
    pobs : ndarray((T,N), dtype = float)
        pobs[t,i] is the observation probability for observation at time t given hidden state i
    beta_out : ndarray((T,N), dtype = float), optional, default = None
        containter for the beta result variables. If None, a new container will be created.
    dtype : type, optional, default = np.float32
        data type of the result.

    Returns
    -------
    beta : ndarray((T,N), dtype = float), optional, default = None
        beta[t,i] is the ith backward coefficient of time t. These can be
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
    if beta_out is None:
        beta_out = np.zeros((T, N), dtype=dtype)
    elif T > beta_out.shape[0]:
        raise ValueError('beta_out must at least have length T in order to fit trajectory.')

    # initialization
    beta_out[T-1, :] = 1.0
    # scaling factor
    scale = np.sum(beta_out[T-1, :])
    # scale
    beta_out[T-1, :] /= scale

    # induction
    for t in range(T-2, -1, -1):
        # beta_i(t) = sum_j A_i,j * beta_j(t+1) * B_j,ob(t+1)
        np.dot(A, beta_out[t+1, :] * pobs[t+1, :], out=beta_out[t, :])
        # scaling factor
        scale = np.sum(beta_out[t, :])
        # scale
        beta_out[t, :] /= scale
    return beta_out