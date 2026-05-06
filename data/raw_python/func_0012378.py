def plot_bs_dists(run, fthetas, axes, **kwargs):
    """Helper function for plotting uncertainties on posterior distributions
    using bootstrap resamples and the fgivenx module. Used by bs_param_dists
    and param_logx_diagram.

    Parameters
    ----------
    run: dict
        Nested sampling run to plot.
    fthetas: list of functions
        Quantities to plot. Each must map a 2d theta array to 1d ftheta array -
        i.e. map every sample's theta vector (every row) to a scalar quantity.
        E.g. use lambda x: x[:, 0] to plot the first parameter.
    axes: list of matplotlib axis objects
    ftheta_lims: list, optional
        Plot limits for each ftheta.
    n_simulate: int, optional
        Number of bootstrap replications to use for the fgivenx
        distributions.
    colormap: matplotlib colormap
        Colors to plot fgivenx distribution.
    mean_color: matplotlib color as str
        Color to plot mean of each parameter. If None (default) means are not
        plotted.
    nx: int, optional
        Size of x-axis grid for fgivenx plots.
    ny: int, optional
        Size of y-axis grid for fgivenx plots.
    cache: str or None
        Root for fgivenx caching (no caching if None).
    parallel: bool, optional
        fgivenx parallel option.
    rasterize_contours: bool, optional
        fgivenx rasterize_contours option.
    smooth: bool, optional
        fgivenx smooth option.
    flip_axes: bool, optional
        Whether or not plot should be rotated 90 degrees anticlockwise onto its
        side.
    tqdm_kwargs: dict, optional
        Keyword arguments to pass to the tqdm progress bar when it is used in
        fgivenx while plotting contours.

    Returns
    -------
    cbar: matplotlib colorbar
        For use in higher order functions.
    """
    ftheta_lims = kwargs.pop('ftheta_lims', [[-1, 1]] * len(fthetas))
    n_simulate = kwargs.pop('n_simulate', 100)
    colormap = kwargs.pop('colormap', plt.get_cmap('Reds_r'))
    mean_color = kwargs.pop('mean_color', None)
    nx = kwargs.pop('nx', 100)
    ny = kwargs.pop('ny', nx)
    cache_in = kwargs.pop('cache', None)
    parallel = kwargs.pop('parallel', True)
    rasterize_contours = kwargs.pop('rasterize_contours', True)
    smooth = kwargs.pop('smooth', False)
    flip_axes = kwargs.pop('flip_axes', False)
    tqdm_kwargs = kwargs.pop('tqdm_kwargs', {'leave': False})
    if kwargs:
        raise TypeError('Unexpected **kwargs: {0}'.format(kwargs))
    assert len(fthetas) == len(axes), \
        'There should be the same number of axes and functions to plot'
    assert len(fthetas) == len(ftheta_lims), \
        'There should be the same number of axes and functions to plot'
    threads = nestcheck.ns_run_utils.get_run_threads(run)
    # get a list of evenly weighted theta samples from bootstrap resampling
    bs_samps = []
    for i in range(n_simulate):
        run_temp = nestcheck.error_analysis.bootstrap_resample_run(
            run, threads=threads)
        w_temp = nestcheck.ns_run_utils.get_w_rel(run_temp, simulate=False)
        bs_samps.append((run_temp['theta'], w_temp))
    for nf, ftheta in enumerate(fthetas):
        # Make an array where each row contains one bootstrap replication's
        # samples
        max_samps = 2 * max([bs_samp[0].shape[0] for bs_samp in bs_samps])
        samples_array = np.full((n_simulate, max_samps), np.nan)
        for i, (theta, weights) in enumerate(bs_samps):
            nsamp = 2 * theta.shape[0]
            samples_array[i, :nsamp:2] = ftheta(theta)
            samples_array[i, 1:nsamp:2] = weights
        ftheta_vals = np.linspace(ftheta_lims[nf][0], ftheta_lims[nf][1], nx)
        try:
            cache = cache_in + '_' + str(nf)
        except TypeError:
            cache = None
        samp_kde = functools.partial(alternate_helper,
                                     func=weighted_1d_gaussian_kde)
        y, pmf = fgivenx.drivers.compute_pmf(
            samp_kde, ftheta_vals, samples_array, ny=ny, cache=cache,
            parallel=parallel, tqdm_kwargs=tqdm_kwargs)
        if flip_axes:
            cbar = fgivenx.plot.plot(
                y, ftheta_vals, np.swapaxes(pmf, 0, 1), axes[nf],
                colors=colormap, rasterize_contours=rasterize_contours,
                smooth=smooth)
        else:
            cbar = fgivenx.plot.plot(
                ftheta_vals, y, pmf, axes[nf], colors=colormap,
                rasterize_contours=rasterize_contours, smooth=smooth)
    # Plot means
    # ----------
    if mean_color is not None:
        w_rel = nestcheck.ns_run_utils.get_w_rel(run, simulate=False)
        w_rel /= np.sum(w_rel)
        means = [np.sum(w_rel * f(run['theta'])) for f in fthetas]
        for nf, mean in enumerate(means):
            if flip_axes:
                axes[nf].axhline(y=mean, lw=1, linestyle='--',
                                 color=mean_color)
            else:
                axes[nf].axvline(x=mean, lw=1, linestyle='--',
                                 color=mean_color)
    return cbar