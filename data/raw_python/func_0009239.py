def u_product(a, b):
    r"""
    Inner product in the Hilbert space of :math:`U`-centered distance matrices.

    This inner product is defined as

    .. math::
        \frac{1}{n(n-3)} \sum_{i,j=1}^n a_{i, j} b_{i, j}

    Parameters
    ----------
    a: array_like
        First input array to be multiplied.
    b: array_like
        Second input array to be multiplied.

    Returns
    -------
    numpy scalar
        Inner product.

    See Also
    --------
    mean_product

    Examples
    --------
    >>> import numpy as np
    >>> import dcor
    >>> a = np.array([[  0.,   3.,  11.,   6.],
    ...               [  3.,   0.,   8.,   3.],
    ...               [ 11.,   8.,   0.,   5.],
    ...               [  6.,   3.,   5.,   0.]])
    >>> b = np.array([[  0.,  13.,  11.,   3.],
    ...               [ 13.,   0.,   2.,  10.],
    ...               [ 11.,   2.,   0.,   8.],
    ...               [  3.,  10.,   8.,   0.]])
    >>> u_a = dcor.u_centered(a)
    >>> u_a
    array([[ 0., -2.,  1.,  1.],
           [-2.,  0.,  1.,  1.],
           [ 1.,  1.,  0., -2.],
           [ 1.,  1., -2.,  0.]])
    >>> u_b = dcor.u_centered(b)
    >>> u_b
    array([[ 0.        ,  2.66666667,  2.66666667, -5.33333333],
           [ 2.66666667,  0.        , -5.33333333,  2.66666667],
           [ 2.66666667, -5.33333333,  0.        ,  2.66666667],
           [-5.33333333,  2.66666667,  2.66666667,  0.        ]])
    >>> dcor.u_product(u_a, u_a)
    6.0
    >>> dcor.u_product(u_a, u_b)
    -8.0

    Note that the formula is well defined as long as the matrices involved
    are square and have the same dimensions, even if they are not in the
    Hilbert space of :math:`U`-centered distance matrices

    >>> dcor.u_product(a, a)
    132.0

    Also the formula produces a division by 0 for 3x3 matrices

    >>> import warnings
    >>> b = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     dcor.u_product(b, b)
    inf

    """
    n = np.size(a, 0)

    return np.sum(a * b) / (n * (n - 3))