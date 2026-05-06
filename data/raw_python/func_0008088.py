def regularize_hidden(p0, P, reversible=True, stationary=False, C=None, eps=None):
    """ Regularizes the hidden initial distribution and transition matrix.

    Makes sure that the hidden initial distribution and transition matrix have
    nonzero probabilities by setting them to eps and then renormalizing.
    Avoids zeros that would cause estimation algorithms to crash or get stuck
    in suboptimal states.

    Parameters
    ----------
    p0 : ndarray(n)
        Initial hidden distribution of the HMM
    P : ndarray(n, n)
        Hidden transition matrix
    reversible : bool
        HMM is reversible. Will make sure it is still reversible after modification.
    stationary : bool
        p0 is the stationary distribution of P. In this case, will not regularize
        p0 separately. If stationary=False, the regularization will be applied to p0.
    C : ndarray(n, n)
        Hidden count matrix. Only needed for stationary=True and P disconnected.
    epsilon : float or None
        minimum value of the resulting transition matrix. Default: evaluates
        to 0.01 / n. The coarse-graining equation can lead to negative elements
        and thus epsilon should be set to at least 0. Positive settings of epsilon
        are similar to a prior and enforce minimum positive values for all
        transition probabilities.

    Return
    ------
    p0 : ndarray(n)
        regularized initial distribution
    P : ndarray(n, n)
        regularized transition matrix

    """
    # input
    n = P.shape[0]
    if eps is None:  # default output probability, in order to avoid zero columns
        eps = 0.01 / n

    # REGULARIZE P
    P = np.maximum(P, eps)
    # and renormalize
    P /= P.sum(axis=1)[:, None]
    # ensure reversibility
    if reversible:
        P = _tmatrix_disconnected.enforce_reversible_on_closed(P)

    # REGULARIZE p0
    if stationary:
        _tmatrix_disconnected.stationary_distribution(P, C=C)
    else:
        p0 = np.maximum(p0, eps)
        p0 /= p0.sum()

    return p0, P