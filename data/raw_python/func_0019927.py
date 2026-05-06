def cos(x):
    """
    Cosine
    """
    if isinstance(x, UncertainFunction):
        mcpts = np.cos(x._mcpts)
        return UncertainFunction(mcpts)
    else:
        return np.cos(x)