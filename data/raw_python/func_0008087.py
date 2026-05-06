def coarse_grain_transition_matrix(P, M):
    """ Coarse grain transition matrix P using memberships M

    Computes

    .. math:
        Pc = (M' M)^-1 M' P M

    Parameters
    ----------
    P : ndarray(n, n)
        microstate transition matrix
    M : ndarray(n, m)
        membership matrix. Membership to macrostate m for each microstate.

    Returns
    -------
    Pc : ndarray(m, m)
        coarse-grained transition matrix.

    """
    # coarse-grain matrix: Pc = (M' M)^-1 M' P M
    W = np.linalg.inv(np.dot(M.T, M))
    A = np.dot(np.dot(M.T, P), M)
    P_coarse = np.dot(W, A)

    # this coarse-graining can lead to negative elements. Setting them to zero here.
    P_coarse = np.maximum(P_coarse, 0)
    # and renormalize
    P_coarse /= P_coarse.sum(axis=1)[:, None]

    return P_coarse