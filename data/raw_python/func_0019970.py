def covariance_matrix(nums_with_uncert):
    """
    Calculate the covariance matrix of uncertain variables, oriented by the
    order of the inputs
    
    Parameters
    ----------
    nums_with_uncert : array-like
        A list of variables that have an associated uncertainty
    
    Returns
    -------
    cov_matrix : 2d-array-like
        A nested list containing covariance values
    
    Example
    -------
    
        >>> x = N(1, 0.1)
        >>> y = N(10, 0.1)
        >>> z = x + 2*y
        >>> covariance_matrix([x,y,z])
        [[  9.99694861e-03   2.54000840e-05   1.00477488e-02]
         [  2.54000840e-05   9.99823207e-03   2.00218642e-02]
         [  1.00477488e-02   2.00218642e-02   5.00914772e-02]]

    """
    ufuncs = list(map(to_uncertain_func, nums_with_uncert))
    cov_matrix = []
    for (i1, expr1) in enumerate(ufuncs):
        coefs_expr1 = []
        mean1 = expr1.mean
        for (i2, expr2) in enumerate(ufuncs[: i1 + 1]):
            mean2 = expr2.mean
            coef = np.mean((expr1._mcpts - mean1) * (expr2._mcpts - mean2))
            coefs_expr1.append(coef)
        cov_matrix.append(coefs_expr1)

    # We symmetrize the matrix:
    for (i, covariance_coefs) in enumerate(cov_matrix):
        covariance_coefs.extend(cov_matrix[j][i] for j in range(i + 1, len(cov_matrix)))

    return cov_matrix