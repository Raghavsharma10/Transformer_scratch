def exp(x):
    """
    Exponential function
    """
    if isinstance(x, UncertainFunction):
        mcpts = np.exp(x._mcpts)
        return UncertainFunction(mcpts)
    else:
        return np.exp(x)