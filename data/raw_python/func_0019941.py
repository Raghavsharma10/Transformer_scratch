def sqrt(x):
    """
    Square-root function
    """
    if isinstance(x, UncertainFunction):
        mcpts = np.sqrt(x._mcpts)
        return UncertainFunction(mcpts)
    else:
        return np.sqrt(x)