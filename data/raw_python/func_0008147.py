def estimate_hmm(observations, nstates, lag=1, initial_model=None, output=None,
                 reversible=True, stationary=False, p=None, accuracy=1e-3, maxit=1000, maxit_P=100000,
                 mincount_connectivity=1e-2):
    r""" Estimate maximum-likelihood HMM

    Generic maximum-likelihood estimation of HMMs

    Parameters
    ----------
    observations : list of numpy arrays representing temporal data
        `observations[i]` is a 1d numpy array corresponding to the observed trajectory index `i`
    nstates : int
        The number of states in the model.
    lag : int
        the lag time at which observations should be read
    initial_model : HMM, optional, default=None
        If specified, the given initial model will be used to initialize the BHMM.
        Otherwise, a heuristic scheme is used to generate an initial guess.
    output : str, optional, default=None
        Output model type from [None, 'gaussian', 'discrete']. If None, will automatically select an output
        model type based on the format of observations.
    reversible : bool, optional, default=True
        If True, a prior that enforces reversible transition matrices (detailed balance) is used;
        otherwise, a standard  non-reversible prior is used.
    stationary : bool, optional, default=False
        If True, the initial distribution of hidden states is self-consistently computed as the stationary
        distribution of the transition matrix. If False, it will be estimated from the starting states.
        Only set this to true if you're sure that the observation trajectories are initiated from a global
        equilibrium distribution.
    p : ndarray (nstates), optional, default=None
        Initial or fixed stationary distribution. If given and stationary=True, transition matrices will be
        estimated with the constraint that they have p as their stationary distribution. If given and
        stationary=False, p is the fixed initial distribution of hidden states.
    accuracy : float
        convergence threshold for EM iteration. When two the likelihood does not increase by more than accuracy, the
        iteration is stopped successfully.
    maxit : int
        stopping criterion for EM iteration. When so many iterations are performed without reaching the requested
        accuracy, the iteration is stopped without convergence (a warning is given)

    Return
    ------
    hmm : :class:`HMM <bhmm.hmm.generic_hmm.HMM>`

    """
    # select output model type
    if output is None:
        output = _guess_output_type(observations)

    if lag > 1:
        observations = lag_observations(observations, lag)

    # construct estimator
    from bhmm.estimators.maximum_likelihood import MaximumLikelihoodEstimator as _MaximumLikelihoodEstimator
    est = _MaximumLikelihoodEstimator(observations, nstates, initial_model=initial_model, output=output,
                                      reversible=reversible, stationary=stationary, p=p, accuracy=accuracy,
                                      maxit=maxit, maxit_P=maxit_P)
    # run
    est.fit()
    # set lag time
    est.hmm._lag = lag
    # return model
    # TODO: package into specific class (DiscreteHMM, GaussianHMM)
    return est.hmm