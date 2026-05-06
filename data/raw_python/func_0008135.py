def enforce_reversible_on_closed(P):
    """ Enforces transition matrix P to be reversible on its closed sets. """
    import msmtools.analysis as msmana
    n = np.shape(P)[0]
    Prev = P.copy()
    # treat each weakly connected set separately
    sets = closed_sets(P)
    for s in sets:
        I = np.ix_(s, s)
        # compute stationary probability
        pi_s = msmana.stationary_distribution(P[I])
        # symmetrize
        X_s = pi_s[:, None] * P[I]
        X_s = 0.5 * (X_s + X_s.T)
        # normalize
        Prev[I] = X_s / X_s.sum(axis=1)[:, None]
    return Prev