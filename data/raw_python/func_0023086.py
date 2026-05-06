def log2_lut(v):
    """
    See `this algo <https://graphics.stanford.edu/~seander/bithacks.html#IntegerLogLookup>`__ for
    computing the log2 of a 32 bit integer using a look up table

    Parameters
    ----------
    v : int
        32 bit integer

    Returns
    -------

    """
    res = np.zeros(v.shape, dtype=np.int32)

    tt = v >> 16
    tt_zero = (tt == 0)
    tt_not_zero = ~tt_zero

    t_h = tt >> 8
    t_zero_h = (t_h == 0) & tt_not_zero
    t_not_zero_h = ~t_zero_h & tt_not_zero

    res[t_zero_h] = LogTable256[tt[t_zero_h]] + 16
    res[t_not_zero_h] = LogTable256[t_h[t_not_zero_h]] + 24

    t_l = v >> 8
    t_zero_l = (t_l == 0) & tt_zero
    t_not_zero_l = ~t_zero_l & tt_zero

    res[t_zero_l] = LogTable256[v[t_zero_l]]
    res[t_not_zero_l] = LogTable256[t_l[t_not_zero_l]] + 8

    return res