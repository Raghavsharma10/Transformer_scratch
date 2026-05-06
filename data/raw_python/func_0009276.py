def pairwise_distances(x, y=None, **kwargs):
    r"""
    pairwise_distances(x, y=None, *, exponent=1)

    Pairwise distance between points.

    Return the pairwise distance between points in two sets, or
    in the same set if only one set is passed.

    Parameters
    ----------
    x: array_like
        An :math:`n \times m` array of :math:`n` observations in
        a :math:`m`-dimensional space.
    y: array_like
        An :math:`l \times m` array of :math:`l` observations in
        a :math:`m`-dimensional space. If None, the distances will
        be computed between the points in :math:`x`.
    exponent: float
        Exponent of the Euclidean distance.

    Returns
    -------
    numpy ndarray
        A :math:`n \times l` matrix where the :math:`(i, j)`-th entry is the
        distance between :math:`x[i]` and :math:`y[j]`.

    Examples
    --------
    >>> import numpy as np
    >>> import dcor
    >>> a = np.array([[1, 2, 3, 4],
    ...               [5, 6, 7, 8],
    ...               [9, 10, 11, 12],
    ...               [13, 14, 15, 16]])
    >>> b = np.array([[16, 15, 14, 13],
    ...               [12, 11, 10, 9],
    ...               [8, 7, 6, 5],
    ...               [4, 3, 2, 1]])
    >>> dcor.distances.pairwise_distances(a)
    array([[ 0.,  8., 16., 24.],
           [ 8.,  0.,  8., 16.],
           [16.,  8.,  0.,  8.],
           [24., 16.,  8.,  0.]])
    >>> dcor.distances.pairwise_distances(a, b)
    array([[24.41311123, 16.61324773,  9.16515139,  4.47213595],
           [16.61324773,  9.16515139,  4.47213595,  9.16515139],
           [ 9.16515139,  4.47213595,  9.16515139, 16.61324773],
           [ 4.47213595,  9.16515139, 16.61324773, 24.41311123]])

    """
    x = _transform_to_2d(x)

    if y is None or y is x:
        return _pdist(x, **kwargs)
    else:
        y = _transform_to_2d(y)
        return _cdist(x, y, **kwargs)