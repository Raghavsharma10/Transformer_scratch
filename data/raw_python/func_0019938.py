def radians(x):
    """
    Convert degrees to radians
    """
    if isinstance(x, UncertainFunction):
        mcpts = np.radians(x._mcpts)
        return UncertainFunction(mcpts)
    else:
        return np.radians(x)