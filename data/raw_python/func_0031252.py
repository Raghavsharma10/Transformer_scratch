def filter_variance(matrix, top):
    """Filter genes in an expression matrix by variance.

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
    assert isinstance(top, (int, np.integer))

    if top >= matrix.p:
        logger.warning('Variance filter has no effect '
                       '("top" parameter is >= number of genes).')
        return matrix.copy()

    var = np.var(matrix.X, axis=1, ddof=1)
    total_var = np.sum(var)  # total sum of variance
    a = np.argsort(var)
    a = a[::-1]
    sel = np.zeros(matrix.p, dtype=np.bool_)
    sel[a[:top]] = True

    lost_p = matrix.p - top
    lost_var = total_var - np.sum(var[sel])
    logger.info('Selected the %d most variable genes '
                '(excluded %.1f%% of genes, representing %.1f%% '
                'of total variance).',
                top, 100 * (lost_p / float(matrix.p)),
                100 * (lost_var / total_var))

    matrix = matrix.loc[sel]
    return matrix