def steady_state(P):
    """
    Calculates the steady state probability vector for a regular Markov
    transition matrix P.

    Parameters
    ----------
    P        : array
               (k, k), an ergodic Markov transition probability matrix.

    Returns
    -------
             : array
               (k, ), steady state distribution.

    Examples
    --------
    Taken from :cite:`Kemeny1967`. Land of Oz example where the states are
    Rain, Nice and Snow, so there is 25 percent chance that if it
    rained in Oz today, it will snow tomorrow, while if it snowed today in
    Oz there is a 50 percent chance of snow again tomorrow and a 25
    percent chance of a nice day (nice, like when the witch with the monkeys
    is melting).

    >>> import numpy as np
    >>> from giddy.ergodic import steady_state
    >>> p=np.array([[.5, .25, .25],[.5,0,.5],[.25,.25,.5]])
    >>> steady_state(p)
    array([0.4, 0.2, 0.4])

    Thus, the long run distribution for Oz is to have 40 percent of the
    days classified as Rain, 20 percent as Nice, and 40 percent as Snow
    (states are mutually exclusive).

    """

    v, d = la.eig(np.transpose(P))
    d = np.array(d)

    # for a regular P maximum eigenvalue will be 1
    mv = max(v)
    # find its position
    i = v.tolist().index(mv)

    row = abs(d[:, i])

    # normalize eigenvector corresponding to the eigenvalue 1
    return row / sum(row)