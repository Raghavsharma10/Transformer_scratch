def sinh(x):
    """
    Hyperbolic sine
    """
    if isinstance(x, UncertainFunction):
        mcpts = np.sinh(x._mcpts)
        return UncertainFunction(mcpts)
    else:
        return np.sinh(x)