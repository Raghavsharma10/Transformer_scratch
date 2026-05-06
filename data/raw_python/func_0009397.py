def prais(pmat):
    """
    Prais conditional mobility measure.

    Parameters
    ----------
    pmat : matrix
           (k, k), Markov probability transition matrix.

    Returns
    -------
    pr   : matrix
           (1, k), conditional mobility measures for each of the k classes.

    Notes
    -----
    Prais' conditional mobility measure for a class is defined as:

    .. math::

            pr_i = 1 -  p_{i,i}

    Examples
    --------
    >>> import numpy as np
    >>> import libpysal
    >>> from giddy.markov import Markov,prais
    >>> f = libpysal.io.open(libpysal.examples.get_path("usjoin.csv"))
    >>> pci = np.array([f.by_col[str(y)] for y in range(1929,2010)])
    >>> q5 = np.array([mc.Quantiles(y).yb for y in pci]).transpose()
    >>> m = Markov(q5)
    >>> m.transitions
    array([[729.,  71.,   1.,   0.,   0.],
           [ 72., 567.,  80.,   3.,   0.],
           [  0.,  81., 631.,  86.,   2.],
           [  0.,   3.,  86., 573.,  56.],
           [  0.,   0.,   1.,  57., 741.]])
    >>> m.p
    array([[0.91011236, 0.0886392 , 0.00124844, 0.        , 0.        ],
           [0.09972299, 0.78531856, 0.11080332, 0.00415512, 0.        ],
           [0.        , 0.10125   , 0.78875   , 0.1075    , 0.0025    ],
           [0.        , 0.00417827, 0.11977716, 0.79805014, 0.07799443],
           [0.        , 0.        , 0.00125156, 0.07133917, 0.92740926]])
    >>> prais(m.p)
    array([0.08988764, 0.21468144, 0.21125   , 0.20194986, 0.07259074])

    """
    pmat = np.array(pmat)
    pr = 1 - np.diag(pmat)
    return pr