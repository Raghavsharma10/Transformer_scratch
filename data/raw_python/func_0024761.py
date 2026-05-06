def _slow_calcbinflux(len_binwave, i_beg, i_end, avflux, deltaw):
    """Python implementation of ``calcbinflux``.

    This is only used if ``synphot.synphot_utils`` C-extension
    import fails.

    See docstrings.py

    """
    binflux = np.empty(shape=(len_binwave, ), dtype=np.float64)
    intwave = np.empty(shape=(len_binwave, ), dtype=np.float64)

    # Note that, like all Python striding, the range over which
    # we integrate is [first:last).
    for i in range(len(i_beg)):
        first = i_beg[i]
        last = i_end[i]
        cur_dw = deltaw[first:last]
        intwave[i] = cur_dw.sum()
        binflux[i] = np.sum(avflux[first:last] * cur_dw) / intwave[i]

    return binflux, intwave