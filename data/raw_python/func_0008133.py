def estimate_P(C, reversible=True, fixed_statdist=None, maxiter=1000000, maxerr=1e-8, mincount_connectivity=0):
    """ Estimates full transition matrix for general connectivity structure

    Parameters
    ----------
    C : ndarray
        count matrix
    reversible : bool
        estimate reversible?
    fixed_statdist : ndarray or None
        estimate with given stationary distribution
    maxiter : int
        Maximum number of reversible iterations.
    maxerr : float
        Stopping criterion for reversible iteration: Will stop when infinity
        norm  of difference vector of two subsequent equilibrium distributions
        is below maxerr.
    mincount_connectivity : float
        Minimum count which counts as a connection.

    """
    import msmtools.estimation as msmest
    n = np.shape(C)[0]
    # output matrix. Set initially to Identity matrix in order to handle empty states
    P = np.eye(n, dtype=np.float64)
    # decide if we need to proceed by weakly or strongly connected sets
    if reversible and fixed_statdist is None:  # reversible to unknown eq. dist. - use strongly connected sets.
        S = connected_sets(C, mincount_connectivity=mincount_connectivity, strong=True)
        for s in S:
            mask = np.zeros(n, dtype=bool)
            mask[s] = True
            if C[np.ix_(mask, ~mask)].sum() > np.finfo(C.dtype).eps:  # outgoing transitions - use partial rev algo.
                transition_matrix_partial_rev(C, P, mask, maxiter=maxiter, maxerr=maxerr)
            else:  # closed set - use standard estimator
                I = np.ix_(mask, mask)
                if s.size > 1:  # leave diagonal 1 if single closed state.
                    P[I] = msmest.transition_matrix(C[I], reversible=True, warn_not_converged=False,
                                                    maxiter=maxiter, maxerr=maxerr)
    else:  # nonreversible or given equilibrium distribution - weakly connected sets
        S = connected_sets(C, mincount_connectivity=mincount_connectivity, strong=False)
        for s in S:
            I = np.ix_(s, s)
            if not reversible:
                Csub = C[I]
                # any zero rows? must set Cii = 1 to avoid dividing by zero
                zero_rows = np.where(Csub.sum(axis=1) == 0)[0]
                Csub[zero_rows, zero_rows] = 1.0
                P[I] = msmest.transition_matrix(Csub, reversible=False)
            elif reversible and fixed_statdist is not None:
                P[I] = msmest.transition_matrix(C[I], reversible=True, fixed_statdist=fixed_statdist,
                                                maxiter=maxiter, maxerr=maxerr)
            else:  # unknown case
                raise NotImplementedError('Transition estimation for the case reversible=' + str(reversible) +
                                          ' fixed_statdist=' + str(fixed_statdist is not None) + ' not implemented.')
    # done
    return P