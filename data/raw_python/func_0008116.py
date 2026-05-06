def state_counts(gamma, T, out=None):
    """ Sum the probabilities of being in state i to time t

    Parameters
    ----------
    gamma : ndarray((T,N), dtype = float), optional, default = None
        gamma[t,i] is the probabilty at time t to be in state i !
    T : int
        number of time steps

    Returns
    -------
    count : numpy.array shape (N)
            count[i] is the summed probabilty to be in state i !

    See Also
    --------
    state_probabilities : to calculate `gamma`

    """
    return np.sum(gamma[0:T], axis=0, out=out)