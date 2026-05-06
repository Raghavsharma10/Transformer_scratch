def atanh(x):
    """
    Inverse hyperbolic tangent
    """
    if isinstance(x, UncertainFunction):
        mcpts = np.arctanh(x._mcpts)
        return UncertainFunction(mcpts)
    else:
        return np.arctanh(x)