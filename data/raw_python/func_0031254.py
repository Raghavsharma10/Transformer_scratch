def filter_percentile(matrix, top, percentile=50):
    """Filter genes in an expression matrix by percentile expression.

    Parameters
    ----------
    matrix: ExpMatrix
        The expression matrix.
    top: int
        The number of genes to retain.
    percentile: int or float, optinonal
        The percentile to use  Defaults to the median (50th percentile).

    Returns
    -------
    ExpMatrix
        The filtered expression matrix.
    """
    assert isinstance(matrix, ExpMatrix)
    assert isinstance(top, int)
    assert isinstance(percentile, (int, float))

    if top >= matrix.p:
        logger.warning('Gene expression filter with `top` parameter that is '
                       ' >= the number of genes!')
        top = matrix.p

    a = np.argsort(np.percentile(matrix.X, percentile, axis=1))
    a = a[::-1]

    sel = np.zeros(matrix.p, dtype=np.bool_)
    sel[a[:top]] = True

    matrix = matrix.loc[sel]
    return matrix