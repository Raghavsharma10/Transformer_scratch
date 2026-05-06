def sojourn_time(p):
    """
    Calculate sojourn time based on a given transition probability matrix.

    Parameters
    ----------
    p        : array
               (k, k), a Markov transition probability matrix.

    Returns
    -------
             : array
               (k, ), sojourn times. Each element is the expected time a Markov
               chain spends in each states before leaving that state.

    Notes
    -----
    Refer to :cite:`Ibe2009` for more details on sojourn times for Markov
    chains.

    Examples
    --------
    >>> from giddy.markov import sojourn_time
    >>> import numpy as np
    >>> p = np.array([[.5, .25, .25], [.5, 0, .5], [.25, .25, .5]])
    >>> sojourn_time(p)
    array([2., 1., 2.])

    """
    p = np.asarray(p)
    pii = p.diagonal()

    if not (1 - pii).all():
        print("Sojourn times are infinite for absorbing states!")
    return 1 / (1 - pii)