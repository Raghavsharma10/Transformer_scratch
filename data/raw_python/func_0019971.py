def correlation_matrix(nums_with_uncert):
    """
    Calculate the correlation matrix of uncertain variables, oriented by the
    order of the inputs
    
    Parameters
    ----------
    nums_with_uncert : array-like
        A list of variables that have an associated uncertainty
    
    Returns
    -------
    corr_matrix : 2d-array-like
        A nested list containing covariance values
    
    Example
    -------
    
        >>> x = N(1, 0.1)
        >>> y = N(10, 0.1)
        >>> z = x + 2*y
        >>> correlation_matrix([x,y,z])
        [[ 0.99969486  0.00254001  0.4489385 ]
         [ 0.00254001  0.99982321  0.89458702]
         [ 0.4489385   0.89458702  1.        ]]

    """
    ufuncs = list(map(to_uncertain_func, nums_with_uncert))
    data = np.vstack([ufunc._mcpts for ufunc in ufuncs])
    return np.corrcoef(data.T, rowvar=0)