def plot_sampler_fingerprint(
        sampler, hyperprior, weights=None, cutoff_weight=None, nbins=None,
        labels=None, burn=0, chain_mask=None, temp_idx=0, points=None,
        plot_samples=False, sample_color='k', point_color=None, point_lw=3,
        title='', rot_x_labels=False, figsize=None
    ):
    """Make a plot of the sampler's "fingerprint": univariate marginal histograms for all hyperparameters.
    
    The hyperparameters are mapped to [0, 1] using
    :py:meth:`hyperprior.elementwise_cdf`, so this can only be used with prior
    distributions which implement this function.
    
    Returns the figure and axis created.
    
    Parameters
    ----------
    sampler : :py:class:`emcee.Sampler` instance or array, (`n_temps`, `n_chains`, `n_samp`, `n_dim`), (`n_chains`, `n_samp`, `n_dim`) or (`n_samp`, `n_dim`)
        The sampler to plot the chains/marginals of. Can also be an array of
        samples which matches the shape of the `chain` attribute that would be
        present in a :py:class:`emcee.Sampler` instance.
    hyperprior : :py:class:`~gptools.utils.JointPrior` instance
        The joint prior distribution for the hyperparameters. Used to map the
        values to [0, 1] so that the hyperparameters can all be shown on the
        same axis.
    weights : array, (`n_temps`, `n_chains`, `n_samp`), (`n_chains`, `n_samp`) or (`n_samp`,), optional
        The weight for each sample. This is useful for post-processing the
        output from MultiNest sampling, for instance.
    cutoff_weight : float, optional
        If `weights` and `cutoff_weight` are present, points with
        `weights < cutoff_weight * weights.max()` will be excluded. Default is
        to plot all points.
    nbins : int or array of int, (`D`,), optional
        The number of bins dividing [0, 1] to use for each histogram. If a
        single int is given, this is used for all of the hyperparameters. If an
        array of ints is given, these are the numbers of bins for each of the
        hyperparameters. The default is to determine the number of bins using
        the Freedman-Diaconis rule.
    labels : array of str, (`D`,), optional
        The labels for each hyperparameter. Default is to use empty strings.
    burn : int, optional
        The number of samples to burn before making the marginal histograms.
        Default is zero (use all samples).
    chain_mask : (index) array, optional
        Mask identifying the chains to keep before plotting, in case there are
        bad chains. Default is to use all chains.
    temp_idx : int, optional
        Index of the temperature to plot when plotting a
        :py:class:`emcee.PTSampler`. Default is 0 (samples from the posterior).
    points : array, (`D`,) or (`N`, `D`), optional
        Array of point(s) to plot as horizontal lines. Default is None.
    plot_samples : bool, optional
        If True, the samples are plotted as horizontal lines. Default is False.
    sample_color : str, optional
        The color to plot the samples in. Default is 'k', meaning black.
    point_color : str or list of str, optional
        The color to plot the individual points in. Default is to loop through
        matplotlib's default color sequence. If a list is provided, it will be
        cycled through.
    point_lw : float, optional
        Line width to use when plotting the individual points.
    title : str, optional
        Title to use for the plot.
    rot_x_labels : bool, optional
        If True, the labels for the x-axis are rotated 90 degrees. Default is
        False (do not rotate labels).
    figsize : 2-tuple, optional
        The figure size to use. Default is to use the matplotlib default.
    """
    try:
        k = sampler.flatchain.shape[-1]
    except AttributeError:
        # Assumes array input is only case where there is no "flatchain" attribute.
        k = sampler.shape[-1]
    # Process the samples:
    if isinstance(sampler, emcee.EnsembleSampler):
        if chain_mask is None:
            chain_mask = scipy.ones(sampler.chain.shape[0], dtype=bool)
        flat_trace = sampler.chain[chain_mask, burn:, :]
        flat_trace = flat_trace.reshape((-1, k))
    elif isinstance(sampler, emcee.PTSampler):
        if chain_mask is None:
            chain_mask = scipy.ones(sampler.nwalkers, dtype=bool)
        flat_trace = sampler.chain[temp_idx, chain_mask, burn:, :]
        flat_trace = flat_trace.reshape((-1, k))
    elif isinstance(sampler, scipy.ndarray):
        if sampler.ndim == 4:
            if chain_mask is None:
                chain_mask = scipy.ones(sampler.shape[1], dtype=bool)
            flat_trace = sampler[temp_idx, chain_mask, burn:, :]
            flat_trace = flat_trace.reshape((-1, k))
            if weights is not None:
                weights = weights[temp_idx, chain_mask, burn:]
                weights = weights.ravel()
        elif sampler.ndim == 3:
            if chain_mask is None:
                chain_mask = scipy.ones(sampler.shape[0], dtype=bool)
            flat_trace = sampler[chain_mask, burn:, :]
            flat_trace = flat_trace.reshape((-1, k))
            if weights is not None:
                weights = weights[chain_mask, burn:]
                weights = weights.ravel()
        elif sampler.ndim == 2:
            flat_trace = sampler[burn:, :]
            flat_trace = flat_trace.reshape((-1, k))
            if weights is not None:
                weights = weights[burn:]
                weights = weights.ravel()
        if cutoff_weight is not None and weights is not None:
            mask = weights >= cutoff_weight * weights.max()
            flat_trace = flat_trace[mask, :]
            weights = weights[mask]
    else:
        raise ValueError("Unknown sampler class: %s" % (type(sampler),))
    
    if labels is None:
        labels = [''] * k
    
    u = scipy.asarray([hyperprior.elementwise_cdf(p) for p in flat_trace], dtype=float).T
    if nbins is None:
        lq, uq = scipy.stats.scoreatpercentile(u, [25, 75], axis=1)
        h = 2.0 * (uq - lq) / u.shape[0]**(1.0 / 3.0)
        n = scipy.asarray(scipy.ceil(1.0 / h), dtype=int)
    else:
        try:
            iter(nbins)
            n = nbins
        except TypeError:
            n = nbins * scipy.ones(u.shape[0])
    
    hist = [scipy.stats.histogram(uv, numbins=nv, defaultlimits=[0, 1], weights=weights) for uv, nv in zip(u, n)]
    max_ct = max([max(h.count) for h in hist])
    min_ct = min([min(h.count) for h in hist])
    
    f = plt.figure(figsize=figsize)
    a = f.add_subplot(1, 1, 1)
    for i, (h, pn) in enumerate(zip(hist, labels)):
        a.imshow(
            scipy.atleast_2d(scipy.asarray(h.count[::-1], dtype=float)).T,
            cmap='gray_r',
            interpolation='nearest',
            vmin=min_ct,
            vmax=max_ct,
            extent=(i, i + 1, 0, 1),
            aspect='auto'
        )
    
    if plot_samples:
        for p in u:
            for i, uv in enumerate(p):
                a.plot([i, i + 1], [uv, uv], sample_color, alpha=0.1)
    
    if points is not None:
        points = scipy.atleast_2d(scipy.asarray(points, dtype=float))
        u_points = [hyperprior.elementwise_cdf(p) for p in points]
        if point_color is None:
            c_cycle = itertools.cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k'])
        else:
            c_cycle = itertools.cycle(scipy.atleast_1d(point_color))
        for p in u_points:
            c = c_cycle.next()
            for i, uv in enumerate(p):
                a.plot([i, i + 1], [uv, uv], color=c, lw=point_lw)
    
    a.set_xlim(0, len(hist))
    a.set_ylim(0, 1)
    a.set_xticks(0.5 + scipy.arange(0, len(hist), dtype=float))
    a.set_xticklabels(labels)
    if rot_x_labels:
        plt.setp(a.xaxis.get_majorticklabels(), rotation=90)
    a.set_xlabel("parameter")
    a.set_ylabel("$u=F_P(p)$")
    a.set_title(title)
    
    return f, a