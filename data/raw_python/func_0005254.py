def linear_least_squares(a, b, residuals=False):
    """
    Return the least-squares solution to a linear matrix equation.
    Solves the equation `a x = b` by computing a vector `x` that
    minimizes the Euclidean 2-norm `|| b - a x ||^2`.  The equation may
    be under-, well-, or over- determined (i.e., the number of
    linearly independent rows of `a` can be less than, equal to, or
    greater than its number of linearly independent columns).  If `a`
    is square and of full rank, then `x` (but for round-off error) is
    the "exact" solution of the equation.
    Parameters
    ----------
    a : (M, N) array_like
        "Coefficient" matrix.
    b : (M,) array_like
        Ordinate or "dependent variable" values.
    residuals : bool
        Compute the residuals associated with the least-squares solution
    Returns
    -------
    x : (M,) ndarray
        Least-squares solution. The shape of `x` depends on the shape of
        `b`.
    residuals : int (Optional)
        Sums of residuals; squared Euclidean 2-norm for each column in
        ``b - a*x``.
    """
    #  Copyright (c) 2013 Alexandre Drouin. All rights reserved.
    #  From https://gist.github.com/aldro61/5889795
    from warnings import warn
#    from scipy.linalg.fblas import dgemm
    from scipy.linalg.blas import dgemm
#    if type(a) != np.ndarray or not a.flags['C_CONTIGUOUS']:
#        warn('Matrix a is not a C-contiguous numpy array. The solver will create a copy, which will result' + \
#             ' in increased memory usage.')
    a = np.asarray(a, order='c')
    i = dgemm(alpha=1.0, a=a.T, b=a.T, trans_b=True)
    x = np.linalg.solve(i, dgemm(alpha=1.0, a=a.T, b=b)).flatten()
    if residuals:
        return x, np.linalg.norm(np.dot(a, x) - b)
    else:
        return x