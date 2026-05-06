def equal_power(arr1, arr2):
    """
    Create an equal power blend of arr1 (fading out) and arr2 (fading in)
    """
    n = N.shape(arr1)[0]
    try:
        channels = N.shape(arr1)[1]
    except:
        channels = 1

    f_in = N.arange(n) / float(n - 1)
    f_out = N.arange(n - 1, -1, -1) / float(n)

    if channels > 1:
        f_in = N.tile(f_in, (channels, 1)).T
        f_out = N.tile(f_out, (channels, 1)).T

    vals = log_factor(f_out) * arr1 + log_factor(f_in) * arr2

    return limiter(vals)