def atan(x):
    """
    Inverse tangent
    """
    if isinstance(x, UncertainFunction):
        mcpts = np.arctan(x._mcpts)
        return UncertainFunction(mcpts)
    else:
        return np.arctan(x)