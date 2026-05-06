def runningstd(t, data, width):
    """Compute the running standard deviation of a time series.

    Returns `t_new`, `std_r`.
    """
    ne = len(t) - width
    t_new = np.zeros(ne)
    std_r = np.zeros(ne)
    for i in range(ne):
        t_new[i] = np.mean(t[i:i+width+1])
        std_r[i] = scipy.stats.nanstd(data[i:i+width+1])
    return t_new, std_r