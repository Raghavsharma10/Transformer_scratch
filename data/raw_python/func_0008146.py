def init_discrete_hmm(observations, nstates, lag=1, reversible=True, stationary=True, regularize=True,
                      method='connect-spectral', separate=None):
    """Use a heuristic scheme to generate an initial model.

    Parameters
    ----------
    observations : list of ndarray((T_i))
        list of arrays of length T_i with observation data
    nstates : int
        The number of states.
    lag : int
        Lag time at which the observations should be counted.
    reversible : bool
        Estimate reversible HMM transition matrix.
    stationary : bool
        p0 is the stationary distribution of P. Currently only reversible=True is implemented
    regularize : bool
        Regularize HMM probabilities to avoid 0's.
    method : str
        * 'lcs-spectral' : Does spectral clustering on the largest connected set
            of observed states.
        * 'connect-spectral' : Uses a weak regularization to connect the weakly
            connected sets and then initializes HMM using spectral clustering on
            the nonempty set.
        * 'spectral' : Uses spectral clustering on the nonempty subsets. Separated
            observed states will end up in separate hidden states. This option is
            only recommended for small observation spaces. Use connect-spectral for
            large observation spaces.
    separate : None or iterable of int
        Force the given set of observed states to stay in a separate hidden state.
        The remaining nstates-1 states will be assigned by a metastable decomposition.

    Examples
    --------

    Generate initial model for a discrete output model.

    >>> import bhmm
    >>> [model, observations, states] = bhmm.testsystems.generate_synthetic_observations(output='discrete')
    >>> initial_model = init_discrete_hmm(observations, model.nstates)

    """
    import msmtools.estimation as msmest
    from bhmm.init.discrete import init_discrete_hmm_spectral
    C = msmest.count_matrix(observations, lag).toarray()
    # regularization
    if regularize:
        eps_A = None
        eps_B = None
    else:
        eps_A = 0
        eps_B = 0
    if not stationary:
        raise NotImplementedError('Discrete-HMM initialization with stationary=False is not yet implemented.')

    if method=='lcs-spectral':
        lcs = msmest.largest_connected_set(C)
        p0, P, B = init_discrete_hmm_spectral(C, nstates, reversible=reversible, stationary=stationary,
                                              active_set=lcs, separate=separate, eps_A=eps_A, eps_B=eps_B)
    elif method=='connect-spectral':
        # make sure we're strongly connected
        C += msmest.prior_neighbor(C, 0.001)
        nonempty = _np.where(C.sum(axis=0) + C.sum(axis=1) > 0)[0]
        C[nonempty, nonempty] = _np.maximum(C[nonempty, nonempty], 0.001)
        p0, P, B = init_discrete_hmm_spectral(C, nstates, reversible=reversible, stationary=stationary,
                                              active_set=nonempty, separate=separate, eps_A=eps_A, eps_B=eps_B)
    elif method=='spectral':
        p0, P, B = init_discrete_hmm_spectral(C, nstates, reversible=reversible, stationary=stationary,
                                              active_set=None, separate=separate, eps_A=eps_A, eps_B=eps_B)
    else:
        raise NotImplementedError('Unknown discrete-HMM initialization method ' + str(method))

    hmm0 = discrete_hmm(p0, P, B)
    hmm0._lag = lag
    return hmm0