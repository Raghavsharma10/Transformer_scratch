def closed_sets(C, mincount_connectivity=0):
    """ Computes the strongly connected closed sets of C """
    n = np.shape(C)[0]
    S = connected_sets(C, mincount_connectivity=mincount_connectivity, strong=True)
    closed = []
    for s in S:
        mask = np.zeros(n, dtype=bool)
        mask[s] = True
        if C[np.ix_(mask, ~mask)].sum() == 0:  # closed set, take it
            closed.append(s)
    return closed