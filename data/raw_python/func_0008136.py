def is_reversible(P):
    """ Returns if P is reversible on its weakly connected sets """
    import msmtools.analysis as msmana
    # treat each weakly connected set separately
    sets = connected_sets(P, strong=False)
    for s in sets:
        Ps = P[s, :][:, s]
        if not msmana.is_transition_matrix(Ps):
            return False  # isn't even a transition matrix!
        pi = msmana.stationary_distribution(Ps)
        X = pi[:, None] * Ps
        if not np.allclose(X, X.T):
            return False
    # survived.
    return True