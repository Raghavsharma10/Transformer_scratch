def shuffle_matrix(X, ids):
    """
    Random permutation of rows and columns of a matrix

    Parameters
    ----------
    X   : array
          (k, k), array to be permutated.
    ids : array
          range (k, ).

    Returns
    -------
    X   : array
          (k, k) with rows and columns randomly shuffled.

    Examples
    --------
    >>> import numpy as np
    >>> from giddy.util import shuffle_matrix
    >>> X=np.arange(16)
    >>> X.shape=(4,4)
    >>> np.random.seed(10)
    >>> shuffle_matrix(X,list(range(4)))
    array([[10,  8, 11,  9],
           [ 2,  0,  3,  1],
           [14, 12, 15, 13],
           [ 6,  4,  7,  5]])

    """
    np.random.shuffle(ids)
    return X[ids, :][:, ids]