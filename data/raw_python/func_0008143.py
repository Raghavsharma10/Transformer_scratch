def discrete_hmm(pi, P, pout):
    """ Initializes a discrete HMM

    Parameters
    ----------
    pi : ndarray(nstates, )
        Initial distribution.
    P : ndarray(nstates,nstates)
        Hidden transition matrix
    pout : ndarray(nstates,nsymbols)
        Output matrix from hidden states to observable symbols
    pi : ndarray(nstates, )
        Fixed initial (if stationary=False) or fixed stationary distribution (if stationary=True).
    stationary : bool, optional, default=True
        If True: initial distribution is equal to stationary distribution of transition matrix
    reversible : bool, optional, default=True
        If True: transition matrix will fulfill detailed balance constraints.

    """
    from bhmm.hmm.discrete_hmm import DiscreteHMM
    from bhmm.output_models.discrete import DiscreteOutputModel
    # initialize output model
    output_model = DiscreteOutputModel(pout)
    # initialize general HMM
    from bhmm.hmm.generic_hmm import HMM as _HMM
    dhmm = _HMM(pi, P, output_model)
    # turn it into a Gaussian HMM
    dhmm = DiscreteHMM(dhmm)
    return dhmm