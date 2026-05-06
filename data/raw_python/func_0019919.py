def abs(x):
    """
    Absolute value
    """
    if isinstance(x, UncertainFunction):
        mcpts = np.abs(x._mcpts)
        return UncertainFunction(mcpts)
    else:
        return np.abs(x)