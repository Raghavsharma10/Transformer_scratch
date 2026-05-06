def transition_matrix_partial_rev(C, P, S, maxiter=1000000, maxerr=1e-8):
    """Maximum likelihood estimation of transition matrix which is reversible on parts

    Partially-reversible estimation of transition matrix. Maximizes the likelihood:

    .. math:
        P_S &=& arg max prod_{S, :} (p_ij)^c_ij \\
        \Pi_S P_{S,S} &=& \Pi_S P_{S,S}

    where the product runs over all elements of the rows S, and detailed balance only
    acts on the block with rows and columns S. :math:`\Pi_S` is the diagonal matrix of
    equilibrium probabilities restricted to set S.

    Note that this formulation

    Parameters
    ----------
    C : ndarray
        full count matrix
    P : ndarray
        full transition matrix to write to. Will overwrite P[S]
    S : ndarray, bool
        boolean selection of reversible set with outgoing transitions
    maxerr : float
        maximum difference in matrix sums between iterations (infinity norm) in order to stop.

    """
    # test input
    assert np.array_equal(C.shape, P.shape)
    # constants
    A = C[S][:, S]
    B = C[S][:, ~S]
    ATA = A + A.T
    countsums = C[S].sum(axis=1)
    # initialize
    X = 0.5 * ATA
    Y = C[S][:, ~S]
    # normalize X, Y
    totalsum = X.sum() + Y.sum()
    X /= totalsum
    Y /= totalsum
    # rowsums
    rowsums = X.sum(axis=1) + Y.sum(axis=1)
    err = 1.0
    it = 0
    while err > maxerr and it < maxiter:
        # update
        d = countsums / rowsums
        X = ATA / (d[:, None] + d)
        Y = B / d[:, None]
        # normalize X, Y
        totalsum = X.sum() + Y.sum()
        X /= totalsum
        Y /= totalsum
        # update sums
        rowsums_new = X.sum(axis=1) + Y.sum(axis=1)
        # compute error
        err = np.max(np.abs(rowsums_new - rowsums))
        # update
        rowsums = rowsums_new
        it += 1
    # write to P
    P[np.ix_(S, S)] = X
    P[np.ix_(S, ~S)] = Y
    P[S] /= P[S].sum(axis=1)[:, None]