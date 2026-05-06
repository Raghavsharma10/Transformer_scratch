def Triangular(low, peak, high, tag=None):
    """
    A triangular random variate
    
    Parameters
    ----------
    low : scalar
        Lower bound of the distribution support
    peak : scalar
        The location of the triangle's peak (low <= peak <= high)
    high : scalar
        Upper bound of the distribution support
    """
    assert low <= peak <= high, 'Triangular "peak" must lie between "low" and "high"'
    low, peak, high = [float(x) for x in [low, peak, high]]
    return uv(
        ss.triang((1.0 * peak - low) / (high - low), loc=low, scale=(high - low)),
        tag=tag,
    )