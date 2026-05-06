def var_fmpt(P):
    """
    Variances of first mean passage times for an ergodic transition
    probability matrix.

    Parameters
    ----------
    P      : array
             (k, k), an ergodic Markov transition probability matrix.

    Returns
    -------
           : array
             (k, k), elements are the variances for the number of intervals
             required for a chain starting in state i to first enter state j.

    Examples
    --------
    >>> import numpy as np
    >>> from giddy.ergodic import var_fmpt
    >>> p=np.array([[.5, .25, .25],[.5,0,.5],[.25,.25,.5]])
    >>> vfm=var_fmpt(p)
    >>> vfm
    array([[ 5.58333333, 12.        ,  6.88888889],
           [ 6.22222222, 12.        ,  6.22222222],
           [ 6.88888889, 12.        ,  5.58333333]])

    Notes
    -----
    Uses formulation (and examples on p. 83) in :cite:`Kemeny1967`.


    """

    P = np.matrix(P)
    A = P ** 1000
    n, k = A.shape
    I = np.identity(k)
    Z = la.inv(I - P + A)
    E = np.ones_like(Z)
    D = np.diag(1. / np.diag(A))
    Zdg = np.diag(np.diag(Z))
    M = (I - Z + E * Zdg) * D
    ZM = Z * M
    ZMdg = np.diag(np.diag(ZM))
    W = M * (2 * Zdg * D - I) + 2 * (ZM - E * ZMdg)
    return np.array(W - np.multiply(M, M))