def find_period(data,
                min_period=0.2, max_period=32.0,
                coarse_precision=1e-5, fine_precision=1e-9,
                periodogram=Lomb_Scargle,
                period_jobs=1):
    """find_period(data, min_period=0.2, max_period=32.0, coarse_precision=1e-5, fine_precision=1e-9, periodogram=Lomb_Scargle, period_jobs=1)

    Returns the period of *data* according to the given *periodogram*,
    searching first with a coarse precision, and then a fine precision.

    **Parameters**

    data : array-like, shape = [n_samples, 2] or [n_samples, 3]
        Array containing columns *time*, *mag*, and (optional) *error*.
    min_period : number
        Minimum period in search-space.
    max_period : number
        Maximum period in search-space.
    coarse_precision : number
        Distance between contiguous frequencies in search-space during first
        sweep.
    fine_precision : number
        Distance between contiguous frequencies in search-space during second
        sweep.
    periodogram : function
        A function with arguments *data*, *precision*, *min_period*,
        *max_period*, and *period_jobs*, and return value *period*.
    period_jobs : int, optional
        Number of simultaneous processes to use while searching (default 1).

    **Returns**

    period : number
        The period of *data*.
    """
    if min_period >= max_period:
        return min_period

    coarse_period = periodogram(data, coarse_precision, min_period, max_period,
                                period_jobs=period_jobs)

    return coarse_period if coarse_precision <= fine_precision else \
        periodogram(data, fine_precision,
                    coarse_period - coarse_precision,
                    coarse_period + coarse_precision,
                    period_jobs=period_jobs)