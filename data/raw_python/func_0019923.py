def asinh(x):
    """
    Inverse hyperbolic sine
    """
    if isinstance(x, UncertainFunction):
        mcpts = np.arcsinh(x._mcpts)
        return UncertainFunction(mcpts)
    else:
        return np.arcsinh(x)