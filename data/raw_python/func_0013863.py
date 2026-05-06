def round_60(value):
    """ round the number to the multiple of 60

    Say a random value is represented by: 60 * n + r
    n is an integer and r is an integer between 0 and 60.
    if r < 30, the result is 60 * n.
    otherwise, the result is 60 * (n + 1)

    The use of this function is that the counter refreshment on
    VNX is always 1 minute.  So the delta time between samples of
    counters must be the multiple of 60.
    :param value: the value to be rounded.
    :return: result
    """
    t = 60
    if value is not None:
        r = value % t
        if r > t / 2:
            ret = value + (t - r)
        else:
            ret = value - r
    else:
        ret = NaN
    return ret