def plot_sampler_cov(
        sampler, method='corr', weights=None, cutoff_weight=None, labels=None,
        burn=0, chain_mask=None, temp_idx=0, cbar_label=None, title='',
        rot_x_labels=False, figsize=None, xlabel_on_top=True
    ):
    """Make a plot of the sampler's correlation or covariance matrix.
    
    Returns the figure and axis created.
    
    Parameters
    ----------
    sampler : :py:class:`emcee.Sampler` instance or array, (`n_temps`, `n_chains`, `n_samp`, `n_dim`), (`n_chains`, `n_samp`, `n_dim`) or (`n_samp`, `n_dim`)
        The sampler to plot the chains/marginals of. Can also be an array of
        samples which matches the shape of the `chain` attribute that would be
        present in a :py:class:`emcee.Sampler` instance.
    method : {'corr', 'cov'}
        Whether to plot the correlation matrix ('corr') or the covariance matrix
        ('cov'). The covariance matrix is often not useful because different
        parameters have wildly different scales. Default is to plot the
        correlation matrix.
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
    cbar_label : str, optional
        The label to use for the colorbar. The default is chosen based on the
        value of the `method` keyword.
    title : str, optional
        Title to use for the plot.
    rot_x_labels : bool, optional
        If True, the labels for the x-axis are rotated 90 degrees. Default is
        False (do not rotate labels).
    figsize : 2-tuple, optional
        The figure size to use. Default is to use the matplotlib default.
    xlabel_on_top : bool, optional
        If True, the x-axis labels are put on top (the way mathematicians
        present matrices). Default is True.
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
    
    if cbar_label is None:
        cbar_label = r'$\mathrm{cov}(p_1, p_2)$' if method == 'cov' else r'$\mathrm{corr}(p_1, p_2)$'
    
    if weights is None:
        if method == 'corr':
            cov = scipy.corrcoef(flat_trace, rowvar=0, ddof=1)
        else:
            cov = scipy.cov(flat_trace, rowvar=0, ddof=1)
    else:
        cov = scipy.cov(flat_trace, rowvar=0, aweights=weights)
        if method == 'corr':
            stds = scipy.sqrt(scipy.diag(cov))
            STD_1, STD_2 = scipy.meshgrid(stds, stds)
            cov = cov / (STD_1 * STD_2)
    
    f_cov = plt.figure(figsize=figsize)
    a_cov = f_cov.add_subplot(1, 1, 1)
    a_cov.set_title(title)
    if method == 'cov':
        vmax = scipy.absolute(cov).max()
    else:
        vmax = 1.0
    cax = a_cov.pcolor(cov, cmap='seismic', vmin=-1 * vmax, vmax=vmax)
    divider = make_axes_locatable(a_cov)
    a_cb = divider.append_axes("right", size="10%", pad=0.05)
    cbar = f_cov.colorbar(cax, cax=a_cb, label=cbar_label)
    a_cov.set_xlabel('parameter')
    a_cov.set_ylabel('parameter')
    a_cov.axis('square')
    a_cov.invert_yaxis()
    if xlabel_on_top:
        a_cov.xaxis.tick_top()
        a_cov.xaxis.set_label_position('top')
    a_cov.set_xticks(0.5 + scipy.arange(0, flat_trace.shape[1], dtype=float))
    a_cov.set_yticks(0.5 + scipy.arange(0, flat_trace.shape[1], dtype=float))
    a_cov.set_xticklabels(labels)
    if rot_x_labels:
        plt.setp(a_cov.xaxis.get_majorticklabels(), rotation=90)
    a_cov.set_yticklabels(labels)
    a_cov.set_xlim(0, flat_trace.shape[1])
    a_cov.set_ylim(flat_trace.shape[1], 0)
    
    return f_cov, a_cov