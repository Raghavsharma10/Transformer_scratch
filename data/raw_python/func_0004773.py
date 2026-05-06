def _weighted_median(values, weights, quantile):
    """ Calculate a weighted median for values above a particular quantile cut

    Used in pseudo continuum normalization

    Parameters
    ----------
    values: np ndarray of floats
        the values to take the median of
    weights: np ndarray of floats
        the weights associated with the values
    quantile: float
        the cut applied to the input data

    Returns
    ------
    the weighted median
    """
    sindx = np.argsort(values)
    cvalues = 1. * np.cumsum(weights[sindx])
    if cvalues[-1] == 0: # means all the values are 0
        return values[0]
    cvalues = cvalues / cvalues[-1] # div by largest value
    foo = sindx[cvalues > quantile]
    if len(foo) == 0:
        return values[0]
    indx = foo[0]
    return values[indx]