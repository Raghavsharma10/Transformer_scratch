def summarize_sampler(sampler, weights=None, burn=0, ci=0.95, chain_mask=None):
    r"""Create summary statistics of the flattened chain of the sampler.
    
    The confidence regions are computed from the quantiles of the data.
    
    Parameters
    ----------
    sampler : :py:class:`emcee.Sampler` instance or array, (`n_temps`, `n_chains`, `n_samp`, `n_dim`), (`n_chains`, `n_samp`, `n_dim`) or (`n_samp`, `n_dim`)
        The sampler to summarize the chains of.
    weights : array, (`n_temps`, `n_chains`, `n_samp`), (`n_chains`, `n_samp`) or (`n_samp`,), optional
        The weight for each sample. This is useful for post-processing the
        output from MultiNest sampling, for instance.
    burn : int, optional
        The number of samples to burn from the beginning of the chain. Default
        is 0 (no burn).
    ci : float, optional
        A number between 0 and 1 indicating the confidence region to compute.
        Default is 0.95 (return upper and lower bounds of the 95% confidence
        interval).
    chain_mask : (index) array, optional
        Mask identifying the chains to keep before plotting, in case there are
        bad chains. Default is to use all chains.
    
    Returns
    -------
    mean : array, (num_params,)
        Mean values of each of the parameters sampled.
    ci_l : array, (num_params,)
        Lower bounds of the `ci*100%` confidence intervals.
    ci_u : array, (num_params,)
        Upper bounds of the `ci*100%` confidence intervals.
    """
    try:
        k = sampler.flatchain.shape[-1]
    except AttributeError:
        # Assumes array input is only case where there is no "flatchain" attribute.
        k = sampler.shape[-1]
    
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
    else:
        raise ValueError("Unknown sampler class: %s" % (type(sampler),))
    
    cibdry = 100.0 * (1.0 - ci) / 2.0
    if weights is None:
        mean = scipy.mean(flat_trace, axis=0)
        ci_l, ci_u = scipy.percentile(flat_trace, [cibdry, 100.0 - cibdry], axis=0)
    else:
        mean = weights.dot(flat_trace) / weights.sum()
        ci_l = scipy.zeros(k)
        ci_u = scipy.zeros(k)
        p = scipy.asarray([cibdry, 100.0 - cibdry])
        for i in range(0, k):
            srt = flat_trace[:, i].argsort()
            x = flat_trace[srt, i]
            w = weights[srt]
            Sn = w.cumsum()
            pn = 100.0 / Sn[-1] * (Sn - w / 2.0)
            j = scipy.digitize(p, pn) - 1
            ci_l[i], ci_u[i] = x[j] + (p - pn[j]) / (pn[j + 1] - pn[j]) * (x[j + 1] - x[j])
    
    return (mean, ci_l, ci_u)