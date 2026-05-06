def nonempty_set(C, mincount_connectivity=0):
    """ Returns the set of states that have at least one incoming or outgoing count """
    # truncate to states with at least one observed incoming or outgoing count.
    if mincount_connectivity > 0:
        C = C.copy()
        C[np.where(C < mincount_connectivity)] = 0
    return np.where(C.sum(axis=0) + C.sum(axis=1) > 0)[0]