def trunc(x):
    """
    Truncate the values to the integer value without rounding
    """
    if isinstance(x, UncertainFunction):
        mcpts = np.trunc(x._mcpts)
        return UncertainFunction(mcpts)
    else:
        return np.trunc(x)