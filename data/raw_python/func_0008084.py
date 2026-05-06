def transition_counts(alpha, beta, A, pobs, T=None, out=None, dtype=np.float32):
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
    dtype : type, optional, default = np.float32
        data type of the result.

    Returns
    -------
    counts : numpy.array shape (N, N)
         counts[i, j] is the summed probability to transition from i to j in time [0,T)

    See Also
    --------
    forward : calculate forward coefficients `alpha`
    backward : calculate backward coefficients `beta`

    """
    # set T
    if T is None:
        T = pobs.shape[0]  # if not set, use the length of pobs as trajectory length
    elif T > pobs.shape[0]:
        raise ValueError('T must be at most the length of pobs.')
    # set N
    N = len(A)
    # output
    if out is None:
        out = np.zeros((N, N), dtype=dtype, order='C')
    else:
        out[:] = 0.0
    # compute transition counts
    xi = np.zeros((N, N), dtype=dtype, order='C')
    for t in range(T-1):
        # xi_i,j(t) = alpha_i(t) * A_i,j * B_j,ob(t+1) * beta_j(t+1)
        np.dot(alpha[t, :][:, None] * A, np.diag(pobs[t+1, :] * beta[t+1, :]), out=xi)
        # normalize to 1 for each time step
        xi /= np.sum(xi)
        # add to counts
        np.add(out, xi, out)
    # return
    return out