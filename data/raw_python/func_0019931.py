def expm1(x):
    """
    Calculate exp(x) - 1
    """
    if isinstance(x, UncertainFunction):
        mcpts = np.expm1(x._mcpts)
        return UncertainFunction(mcpts)
    else:
        return np.expm1(x)