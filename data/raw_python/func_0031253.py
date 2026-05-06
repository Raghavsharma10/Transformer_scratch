def filter_mean(matrix, top):
    """Filter genes in an expression matrix by mean expression.

    Parameters
    ----------
    matrix: ExpMatrix
        The expression matrix.
    top: int
        The number of genes to retain.

    Returns
    -------
    ExpMatrix
        The filtered expression matrix.
    """
    assert isinstance(matrix, ExpMatrix)
    assert isinstance(top, int)

    if top >= matrix.p:
        logger.warning('Gene expression filter with `top` parameter that is '
                       '>= the number of genes!')
        top = matrix.p

    a = np.argsort(np.mean(matrix.X, axis=1))
    a = a[::-1]

    sel = np.zeros(matrix.p, dtype=np.bool_)
    sel[a[:top]] = True

    matrix = matrix.loc[sel]
    return matrix