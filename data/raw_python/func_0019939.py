def sin(x):
    """
    Sine
    """
    if isinstance(x, UncertainFunction):
        mcpts = np.sin(x._mcpts)
        return UncertainFunction(mcpts)
    else:
        return np.sin(x)