def connected_sets(C, mincount_connectivity=0, strong=True):
    """ Computes the connected sets of C.

    C : count matrix
    mincount_connectivity : float
        Minimum count which counts as a connection.
    strong : boolean
        True: Seek strongly connected sets. False: Seek weakly connected sets.

    """
    import msmtools.estimation as msmest
    Cconn = C.copy()
    Cconn[np.where(C <= mincount_connectivity)] = 0
    # treat each connected set separately
    S = msmest.connected_sets(Cconn, directed=strong)
    return S