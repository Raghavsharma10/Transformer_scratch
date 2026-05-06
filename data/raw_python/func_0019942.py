def tan(x):
    """
    Tangent
    """
    if isinstance(x, UncertainFunction):
        mcpts = np.tan(x._mcpts)
        return UncertainFunction(mcpts)
    else:
        return np.tan(x)