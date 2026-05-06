def correlate(params, corrmat):
    """
    Force a correlation matrix on a set of statistically distributed objects.
    This function works on objects in-place.
    
    Parameters
    ----------
    params : array
        An array of of uv objects.
    corrmat : 2d-array
        The correlation matrix to be imposed
    
    """
    # Make sure all inputs are compatible
    assert all(
        [isinstance(param, UncertainFunction) for param in params]
    ), 'All inputs to "correlate" must be of type "UncertainFunction"'

    # Put each ufunc's samples in a column-wise matrix
    data = np.vstack([param._mcpts for param in params]).T

    # Apply the correlation matrix to the sampled data
    new_data = induce_correlations(data, corrmat)

    # Re-set the samples to the respective variables
    for i in range(len(params)):
        params[i]._mcpts = new_data[:, i]