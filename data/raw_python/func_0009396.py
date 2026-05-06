def kullback(F):
    """
    Kullback information based test of Markov Homogeneity.

    Parameters
    ----------
    F : array
        (s, r, r), values are transitions (not probabilities) for
        s strata, r initial states, r terminal states.

    Returns
    -------
    Results : dictionary
              (key - value)

              Conditional homogeneity - (float) test statistic for homogeneity
              of transition probabilities across strata.

              Conditional homogeneity pvalue - (float) p-value for test
              statistic.

              Conditional homogeneity dof - (int) degrees of freedom =
              r(s-1)(r-1).

    Notes
    -----
    Based on :cite:`Kullback1962`.
    Example below is taken from Table 9.2 .

    Examples
    --------
    >>> import numpy as np
    >>> from giddy.markov import kullback
    >>> s1 = np.array([
    ...         [ 22, 11, 24,  2,  2,  7],
    ...         [ 5, 23, 15,  3, 42,  6],
    ...         [ 4, 21, 190, 25, 20, 34],
    ...         [0, 2, 14, 56, 14, 28],
    ...         [32, 15, 20, 10, 56, 14],
    ...         [5, 22, 31, 18, 13, 134]
    ...     ])
    >>> s2 = np.array([
    ...     [3, 6, 9, 3, 0, 8],
    ...     [1, 9, 3, 12, 27, 5],
    ...     [2, 9, 208, 32, 5, 18],
    ...     [0, 14, 32, 108, 40, 40],
    ...     [22, 14, 9, 26, 224, 14],
    ...     [1, 5, 13, 53, 13, 116]
    ...     ])
    >>>
    >>> F = np.array([s1, s2])
    >>> res = kullback(F)
    >>> "%8.3f"%res['Conditional homogeneity']
    ' 160.961'
    >>> "%d"%res['Conditional homogeneity dof']
    '30'
    >>> "%3.1f"%res['Conditional homogeneity pvalue']
    '0.0'

    """

    F1 = F == 0
    F1 = F + F1
    FLF = F * np.log(F1)
    T1 = 2 * FLF.sum()

    FdJK = F.sum(axis=0)
    FdJK1 = FdJK + (FdJK == 0)
    FdJKLFdJK = FdJK * np.log(FdJK1)
    T2 = 2 * FdJKLFdJK.sum()

    FdJd = F.sum(axis=0).sum(axis=1)
    FdJd1 = FdJd + (FdJd == 0)
    T3 = 2 * (FdJd * np.log(FdJd1)).sum()

    FIJd = F[:, :].sum(axis=1)
    FIJd1 = FIJd + (FIJd == 0)
    T4 = 2 * (FIJd * np.log(FIJd1)).sum()

    T6 = F.sum()
    T6 = 2 * T6 * np.log(T6)

    s, r, r1 = F.shape
    chom = T1 - T4 - T2 + T3
    cdof = r * (s - 1) * (r - 1)
    results = {}
    results['Conditional homogeneity'] = chom
    results['Conditional homogeneity dof'] = cdof
    results['Conditional homogeneity pvalue'] = 1 - stats.chi2.cdf(chom, cdof)
    return results