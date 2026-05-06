def chi2(T1, T2):
    """
    chi-squared test of difference between two transition matrices.

    Parameters
    ----------
    T1    : array
            (k, k), matrix of transitions (counts).
    T2    : array
            (k, k), matrix of transitions (counts) to use to form the
            probabilities under the null.

    Returns
    -------
          : tuple
            (3 elements).
            (chi2 value, pvalue, degrees of freedom).

    Examples
    --------
    >>> import libpysal
    >>> from giddy.markov import Spatial_Markov, chi2
    >>> f = libpysal.io.open(libpysal.examples.get_path("usjoin.csv"))
    >>> years = list(range(1929, 2010))
    >>> pci = np.array([f.by_col[str(y)] for y in years]).transpose()
    >>> rpci = pci/(pci.mean(axis=0))
    >>> w = libpysal.io.open(libpysal.examples.get_path("states48.gal")).read()
    >>> w.transform='r'
    >>> sm = Spatial_Markov(rpci, w, fixed=True)
    >>> T1 = sm.T[0]
    >>> T1
    array([[562.,  22.,   1.,   0.],
           [ 12., 201.,  22.,   0.],
           [  0.,  17.,  97.,   4.],
           [  0.,   0.,   3.,  19.]])
    >>> T2 = sm.transitions
    >>> T2
    array([[884.,  77.,   4.,   0.],
           [ 68., 794.,  87.,   3.],
           [  1.,  92., 815.,  51.],
           [  1.,   0.,  60., 903.]])
    >>> chi2(T1,T2)
    (23.39728441473295, 0.005363116704861337, 9)

    Notes
    -----
    Second matrix is used to form the probabilities under the null.
    Marginal sums from first matrix are distributed across these probabilities
    under the null. In other words the observed transitions are taken from T1
    while the expected transitions are formed as follows

    .. math::

            E_{i,j} = \sum_j T1_{i,j} * T2_{i,j}/\sum_j T2_{i,j}

    Degrees of freedom corrected for any rows in either T1 or T2 that have
    zero total transitions.
    """
    rs2 = T2.sum(axis=1)
    rs1 = T1.sum(axis=1)
    rs2nz = rs2 > 0
    rs1nz = rs1 > 0
    dof1 = sum(rs1nz)
    dof2 = sum(rs2nz)
    rs2 = rs2 + (rs2 == 0)
    dof = (dof1 - 1) * (dof2 - 1)
    p = np.diag(1 / rs2) * np.matrix(T2)
    E = np.diag(rs1) * np.matrix(p)
    num = T1 - E
    num = np.multiply(num, num)
    E = E + (E == 0)
    chi2 = num / E
    chi2 = chi2.sum()
    pvalue = 1 - stats.chi2.cdf(chi2, dof)
    return chi2, pvalue, dof