def get_lower(matrix):
    """
    Flattens the lower part of an n x n matrix into an n*(n-1)/2 x 1 vector.

    Parameters
    ----------
    matrix  : array
              (n, n) numpy array, a distance matrix.

    Returns
    -------
    lowvec  : array
              numpy array, the lower half of the distance matrix flattened into
              a vector of length n*(n-1)/2.

    Examples
    --------
    >>> import numpy as np
    >>> from giddy.util import get_lower
    >>> test = np.array([[0,1,2,3],[1,0,1,2],[2,1,0,1],[4,2,1,0]])
    >>> lower = get_lower(test)
    >>> lower
    array([[1],
           [2],
           [1],
           [4],
           [2],
           [1]])

    """
    n = matrix.shape[0]
    lowerlist = []
    for i in range(n):
        for j in range(n):
            if i > j:
                lowerlist.append(matrix[i, j])
    veclen = n * (n - 1) / 2
    lowvec = np.reshape(np.array(lowerlist), (int(veclen), 1))
    return lowvec