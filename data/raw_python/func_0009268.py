def _sqrt(x):
    """
    Return square root of an ndarray.

    This sqrt function for ndarrays tries to use the exponentiation operator
    if the objects stored do not supply a sqrt method.

    """
    x = np.clip(x, a_min=0, a_max=None)

    try:
        return np.sqrt(x)
    except AttributeError:
        exponent = 0.5

        try:
            exponent = np.take(x, 0).from_float(exponent)
        except AttributeError:
            pass

        return x ** exponent