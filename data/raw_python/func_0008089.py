def regularize_pobs(B, nonempty=None, separate=None, eps=None):
    """ Regularizes the output probabilities.

    Makes sure that the output probability distributions has
    nonzero probabilities by setting them to eps and then renormalizing.
    Avoids zeros that would cause estimation algorithms to crash or get stuck
    in suboptimal states.

    Parameters
    ----------
    B : ndarray(n, m)
        HMM output probabilities
    nonempty : None or iterable of int
        Nonempty set. Only regularize on this subset.
    separate : None or iterable of int
        Force the given set of observed states to stay in a separate hidden state.
        The remaining nstates-1 states will be assigned by a metastable decomposition.
    reversible : bool
        HMM is reversible. Will make sure it is still reversible after modification.

    Returns
    -------
    B : ndarray(n, m)
        Regularized output probabilities

    """
    # input
    B = B.copy()  # modify copy
    n, m = B.shape  # number of hidden / observable states
    if eps is None:  # default output probability, in order to avoid zero columns
        eps = 0.01 / m
    # observable sets
    if nonempty is None:
        nonempty = np.arange(m)

    if separate is None:
        B[:, nonempty] = np.maximum(B[:, nonempty], eps)
    else:
        nonempty_nonseparate = np.array(list(set(nonempty) - set(separate)), dtype=int)
        nonempty_separate = np.array(list(set(nonempty).intersection(set(separate))), dtype=int)
        B[:n-1, nonempty_nonseparate] = np.maximum(B[:n-1, nonempty_nonseparate], eps)
        B[n-1, nonempty_separate] = np.maximum(B[n-1, nonempty_separate], eps)

    # renormalize and return copy
    B /= B.sum(axis=1)[:, None]
    return B