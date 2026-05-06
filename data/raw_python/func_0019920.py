def acos(x):
    """
    Inverse cosine
    """
    if isinstance(x, UncertainFunction):
        mcpts = np.arccos(x._mcpts)
        return UncertainFunction(mcpts)
    else:
        return np.arccos(x)