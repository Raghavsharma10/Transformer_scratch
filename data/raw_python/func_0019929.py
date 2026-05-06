def degrees(x):
    """
    Convert radians to degrees
    """
    if isinstance(x, UncertainFunction):
        mcpts = np.degrees(x._mcpts)
        return UncertainFunction(mcpts)
    else:
        return np.degrees(x)