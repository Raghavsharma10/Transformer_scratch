def conditional_entropy(data, precision, min_period, max_period,
                        xbins=10, ybins=5, period_jobs=1):
    """
    Returns the period of *data* by minimizing conditional entropy.
    See `link <http://arxiv.org/pdf/1306.6664v2.pdf>`_ [GDDMD] for details.

    **Parameters**

    data : array-like, shape = [n_samples, 2] or [n_samples, 3]
        Array containing columns *time*, *mag*, and (optional) *error*.
    precision : number
        Distance between contiguous frequencies in search-space.
    min_period : number
        Minimum period in search-space.
    max_period : number
        Maximum period in search-space.
    xbins : int, optional
        Number of phase bins for each trial period (default 10).
    ybins : int, optional
        Number of magnitude bins for each trial period (default 5).
    period_jobs : int, optional
        Number of simultaneous processes to use while searching. Only one
        process will ever be used, but argument is included to conform to
        *periodogram* standards of :func:`find_period` (default 1).

    **Returns**

    period : number
        The period of *data*.

    **Citations**

    .. [GDDMD] Graham, Matthew J. ; Drake, Andrew J. ; Djorgovski, S. G. ;
               Mahabal, Ashish A. ; Donalek, Ciro, 2013,
               Monthly Notices of the Royal Astronomical Society,
               Volume 434, Issue 3, p.2629-2635
    """
    periods = np.arange(min_period, max_period, precision)
    copy = np.ma.copy(data)
    copy[:,1] = (copy[:,1]  - np.min(copy[:,1])) \
       / (np.max(copy[:,1]) - np.min(copy[:,1]))
    partial_job = partial(CE, data=copy, xbins=xbins, ybins=ybins)
    m = map if period_jobs <= 1 else Pool(period_jobs).map
    entropies = list(m(partial_job, periods))

    return periods[np.argmin(entropies)]