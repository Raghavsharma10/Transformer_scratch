def PERT(low, peak, high, g=4.0, tag=None):
    """
    A PERT random variate
    
    Parameters
    ----------
    low : scalar
        Lower bound of the distribution support
    peak : scalar
        The location of the distribution's peak (low <= peak <= high)
    high : scalar
        Upper bound of the distribution support
    
    Optional
    --------
    g : scalar
        Controls the uncertainty of the distribution around the peak. Smaller
        values make the distribution flatter and more uncertain around the 
        peak while larger values make it focused and less uncertain around
        the peak. (Default: 4)
    """
    a, b, c = [float(x) for x in [low, peak, high]]
    assert a <= b <= c, 'PERT "peak" must be greater than "low" and less than "high"'
    assert g >= 0, 'PERT "g" must be non-negative'
    mu = (a + g * b + c) / (g + 2)
    if mu == b:
        a1 = a2 = 3.0
    else:
        a1 = ((mu - a) * (2 * b - a - c)) / ((b - mu) * (c - a))
        a2 = a1 * (c - mu) / (mu - a)

    return Beta(a1, a2, a, c, tag)