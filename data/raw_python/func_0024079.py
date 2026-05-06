def univariate_envelope_plot(x, mean, std, ax=None, base_alpha=0.375, envelopes=[1, 3], lb=None, ub=None, expansion=10, **kwargs):
    """Make a plot of a mean curve with uncertainty envelopes.
    """
    if ax is None:
        f = plt.figure()
        ax = f.add_subplot(1, 1, 1)
    elif ax == 'gca':
        ax = plt.gca()
    
    mean = scipy.asarray(mean, dtype=float).copy()
    std = scipy.asarray(std, dtype=float).copy()
    
    # Truncate the data so matplotlib doesn't die:
    if lb is not None and ub is not None and expansion != 1.0:
        expansion *= ub - lb
        ub = ub + expansion
        lb = lb - expansion
    if ub is not None:
        mean[mean > ub] = ub
    if lb is not None:
        mean[mean < lb] = lb
    
    l = ax.plot(x, mean, **kwargs)
    color = plt.getp(l[0], 'color')
    e = []
    for i in envelopes:
        lower = mean - i * std
        upper = mean + i * std
        if ub is not None:
            lower[lower > ub] = ub
            upper[upper > ub] = ub
        if lb is not None:
            lower[lower < lb] = lb
            upper[upper < lb] = lb
        e.append(ax.fill_between(x, lower, upper, facecolor=color, alpha=base_alpha / i))
    return (l, e)