def gaussian_hmm(pi, P, means, sigmas):
    """ Initializes a 1D-Gaussian HMM

    Parameters
    ----------
    pi : ndarray(nstates, )
        Initial distribution.
    P : ndarray(nstates,nstates)
        Hidden transition matrix
    means : ndarray(nstates, )
        Means of Gaussian output distributions
    sigmas : ndarray(nstates, )
        Standard deviations of Gaussian output distributions
    stationary : bool, optional, default=True
        If True: initial distribution is equal to stationary distribution of transition matrix
    reversible : bool, optional, default=True
        If True: transition matrix will fulfill detailed balance constraints.

    """
    from bhmm.hmm.gaussian_hmm import GaussianHMM
    from bhmm.output_models.gaussian import GaussianOutputModel
    # count states
    nstates = _np.array(P).shape[0]
    # initialize output model
    output_model = GaussianOutputModel(nstates, means, sigmas)
    # initialize general HMM
    from bhmm.hmm.generic_hmm import HMM as _HMM
    ghmm = _HMM(pi, P, output_model)
    # turn it into a Gaussian HMM
    ghmm = GaussianHMM(ghmm)
    return ghmm