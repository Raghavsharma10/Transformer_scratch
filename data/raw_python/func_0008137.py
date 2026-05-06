def stationary_distribution(P, C=None, mincount_connectivity=0):
    """ Simple estimator for stationary distribution for multiple strongly connected sets """
    # can be replaced by msmtools.analysis.stationary_distribution in next msmtools release
    from msmtools.analysis.dense.stationary_vector import stationary_distribution as msmstatdist
    if C is None:
        if is_connected(P, strong=True):
            return msmstatdist(P)
        else:
            raise ValueError('Computing stationary distribution for disconnected matrix. Need count matrix.')

    # disconnected sets
    n = np.shape(C)[0]
    ctot = np.sum(C)
    pi = np.zeros(n)
    # treat each weakly connected set separately
    sets = connected_sets(C, mincount_connectivity=mincount_connectivity, strong=False)
    for s in sets:
        # compute weight
        w = np.sum(C[s, :]) / ctot
        pi[s] = w * msmstatdist(P[s, :][:, s])
    # reinforce normalization
    pi /= np.sum(pi)
    return pi