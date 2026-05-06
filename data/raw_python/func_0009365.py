def fmpt(P):
    """
    Calculates the matrix of first mean passage times for an ergodic transition
    probability matrix.

    Parameters
    ----------
    P    : array
           (k, k), an ergodic Markov transition probability matrix.

    Returns
    -------
    M    : array
           (k, k), elements are the expected value for the number of intervals
           required for a chain starting in state i to first enter state j.
           If i=j then this is the recurrence time.

    Examples
    --------
    >>> import numpy as np
    >>> from giddy.ergodic import fmpt
    >>> p=np.array([[.5, .25, .25],[.5,0,.5],[.25,.25,.5]])
    >>> fm=fmpt(p)
    >>> fm
    array([[2.5       , 4.        , 3.33333333],
           [2.66666667, 5.        , 2.66666667],
           [3.33333333, 4.        , 2.5       ]])

    Thus, if it is raining today in Oz we can expect a nice day to come
    along in another 4 days, on average, and snow to hit in 3.33 days. We can
    expect another rainy day in 2.5 days. If it is nice today in Oz, we would
    experience a change in the weather (either rain or snow) in 2.67 days from
    today. (That wicked witch can only die once so I reckon that is the
    ultimate absorbing state).

    Notes
    -----
    Uses formulation (and examples on p. 218) in :cite:`Kemeny1967`.

    """

    P = np.matrix(P)
    k = P.shape[0]
    A = np.zeros_like(P)
    ss = steady_state(P).reshape(k, 1)
    for i in range(k):
        A[:, i] = ss
    A = A.transpose()
    I = np.identity(k)
    Z = la.inv(I - P + A)
    E = np.ones_like(Z)
    A_diag = np.diag(A)
    A_diag = A_diag + (A_diag == 0)
    D = np.diag(1. / A_diag)
    Zdg = np.diag(np.diag(Z))
    M = (I - Z + E * Zdg) * D
    return np.array(M)