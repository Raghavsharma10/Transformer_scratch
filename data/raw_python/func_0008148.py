def bayesian_hmm(observations, estimated_hmm, nsample=100, reversible=True, stationary=False,
                 p0_prior='mixed', transition_matrix_prior='mixed', store_hidden=False, call_back=None):
    r""" Bayesian HMM based on sampling the posterior

    Generic maximum-likelihood estimation of HMMs

    Parameters
    ----------
    observations : list of numpy arrays representing temporal data
        `observations[i]` is a 1d numpy array corresponding to the observed trajectory index `i`
    estimated_hmm : HMM
        HMM estimated from estimate_hmm or initialize_hmm
    reversible : bool, optional, default=True
        If True, a prior that enforces reversible transition matrices (detailed balance) is used;
        otherwise, a standard  non-reversible prior is used.
    stationary : bool, optional, default=False
        If True, the stationary distribution of the transition matrix will be used as initial distribution.
        Only use True if you are confident that the observation trajectories are started from a global
        equilibrium. If False, the initial distribution will be estimated as usual from the first step
        of the hidden trajectories.
    nsample : int, optional, default=100
        number of Gibbs sampling steps
    p0_prior : None, str, float or ndarray(n)
        Prior for the initial distribution of the HMM. Will only be active
        if stationary=False (stationary=True means that p0 is identical to
        the stationary distribution of the transition matrix).
        Currently implements different versions of the Dirichlet prior that
        is conjugate to the Dirichlet distribution of p0. p0 is sampled from:
        .. math:
            p0 \sim \prod_i (p0)_i^{a_i + n_i - 1}
        where :math:`n_i` are the number of times a hidden trajectory was in
        state :math:`i` at time step 0 and :math:`a_i` is the prior count.
        Following options are available:
        |  'mixed' (default),  :math:`a_i = p_{0,init}`, where :math:`p_{0,init}`
            is the initial distribution of initial_model.
        |  'uniform',  :math:`a_i = 1`
        |  ndarray(n) or float,
            the given array will be used as A.
        |  None,  :math:`a_i = 0`. This option ensures coincidence between
            sample mean an MLE. Will sooner or later lead to sampling problems,
            because as soon as zero trajectories are drawn from a given state,
            the sampler cannot recover and that state will never serve as a starting
            state subsequently. Only recommended in the large data regime and
            when the probability to sample zero trajectories from any state
            is negligible.
    transition_matrix_prior : str or ndarray(n, n)
        Prior for the HMM transition matrix.
        Currently implements Dirichlet priors if reversible=False and reversible
        transition matrix priors as described in [1]_ if reversible=True. For the
        nonreversible case the posterior of transition matrix :math:`P` is:
        .. math:
            P \sim \prod_{i,j} p_{ij}^{b_{ij} + c_{ij} - 1}
        where :math:`c_{ij}` are the number of transitions found for hidden
        trajectories and :math:`b_{ij}` are prior counts.
        |  'mixed' (default),  :math:`b_{ij} = p_{ij,init}`, where :math:`p_{ij,init}`
            is the transition matrix of initial_model. That means one prior
            count will be used per row.
        |  'uniform',  :math:`b_{ij} = 1`
        |  ndarray(n, n) or broadcastable,
            the given array will be used as B.
        |  None,  :math:`b_ij = 0`. This option ensures coincidence between
            sample mean an MLE. Will sooner or later lead to sampling problems,
            because as soon as a transition :math:`ij` will not occur in a
            sample, the sampler cannot recover and that transition will never
            be sampled again. This option is not recommended unless you have
            a small HMM and a lot of data.
    store_hidden : bool, optional, default=False
        store hidden trajectories in sampled HMMs
    call_back : function, optional, default=None
        a call back function with no arguments, which if given is being called
        after each computed sample. This is useful for implementing progress bars.

    Return
    ------
    hmm : :class:`SampledHMM <bhmm.hmm.generic_sampled_hmm.SampledHMM>`

    References
    ----------
    .. [1] Trendelkamp-Schroer, B., H. Wu, F. Paul and F. Noe:
        Estimation and uncertainty of reversible Markov models.
        J. Chem. Phys. 143, 174101 (2015).

    """
    # construct estimator
    from bhmm.estimators.bayesian_sampling import BayesianHMMSampler as _BHMM
    sampler = _BHMM(observations, estimated_hmm.nstates, initial_model=estimated_hmm,
                    reversible=reversible, stationary=stationary, transition_matrix_sampling_steps=1000,
                    p0_prior=p0_prior, transition_matrix_prior=transition_matrix_prior,
                    output=estimated_hmm.output_model.model_type)

    # Sample models.
    sampled_hmms = sampler.sample(nsamples=nsample, save_hidden_state_trajectory=store_hidden,
                                  call_back=call_back)
    # return model
    from bhmm.hmm.generic_sampled_hmm import SampledHMM
    return SampledHMM(estimated_hmm, sampled_hmms)