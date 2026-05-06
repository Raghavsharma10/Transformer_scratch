def asin(x):
    """
    Inverse sine
    """
    if isinstance(x, UncertainFunction):
        mcpts = np.arcsin(x._mcpts)
        return UncertainFunction(mcpts)
    else:
        return np.arcsin(x)