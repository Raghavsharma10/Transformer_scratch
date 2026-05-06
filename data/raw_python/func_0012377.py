def bs_param_dists(run_list, **kwargs):
    """Creates posterior distributions and their bootstrap error functions for
    input runs and estimators.

    For a more detailed description and some example use cases, see 'nestcheck:
    diagnostic tests for nested sampling calculations' (Higson et al. 2019).

    Parameters
    ----------
    run_list: dict or list of dicts
        Nested sampling run(s) to plot.
    fthetas: list of functions, optional
        Quantities to plot. Each must map a 2d theta array to 1d ftheta array -
        i.e. map every sample's theta vector (every row) to a scalar quantity.
        E.g. use lambda x: x[:, 0] to plot the first parameter.
    labels: list of strs, optional
        Labels for each ftheta.
    ftheta_lims: list, optional
        Plot limits for each ftheta.
    n_simulate: int, optional
        Number of bootstrap replications to be used for the fgivenx
        distributions.
    random_seed: int, optional
        Seed to make sure results are consistent and fgivenx caching can be
        used.
    figsize: tuple, optional
        Matplotlib figsize in (inches).
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
    tqdm_kwargs: dict, optional
        Keyword arguments to pass to the tqdm progress bar when it is used in
        fgivenx while plotting contours.

    Returns
    -------
    fig: matplotlib figure
    """
    fthetas = kwargs.pop('fthetas', [lambda theta: theta[:, 0],
                                     lambda theta: theta[:, 1]])
    labels = kwargs.pop('labels', [r'$\theta_' + str(i + 1) + '$' for i in
                                   range(len(fthetas))])
    ftheta_lims = kwargs.pop('ftheta_lims', [[-1, 1]] * len(fthetas))
    n_simulate = kwargs.pop('n_simulate', 100)
    random_seed = kwargs.pop('random_seed', 0)
    figsize = kwargs.pop('figsize', (6.4, 2))
    nx = kwargs.pop('nx', 100)
    ny = kwargs.pop('ny', nx)
    cache_in = kwargs.pop('cache', None)
    parallel = kwargs.pop('parallel', True)
    rasterize_contours = kwargs.pop('rasterize_contours', True)
    tqdm_kwargs = kwargs.pop('tqdm_kwargs', {'disable': True})
    if kwargs:
        raise TypeError('Unexpected **kwargs: {0}'.format(kwargs))
    # Use random seed to make samples consistent and allow caching.
    # To avoid fixing seed use random_seed=None
    state = np.random.get_state()  # save initial random state
    np.random.seed(random_seed)
    if not isinstance(run_list, list):
        run_list = [run_list]
    assert len(labels) == len(fthetas), (
        'There should be the same number of axes and labels')
    width_ratios = [40] * len(fthetas) + [1] * len(run_list)
    fig, axes = plt.subplots(nrows=1, ncols=len(run_list) + len(fthetas),
                             gridspec_kw={'wspace': 0.1,
                                          'width_ratios': width_ratios},
                             figsize=figsize)
    colormaps = ['Reds_r', 'Blues_r', 'Greys_r', 'Greens_r', 'Oranges_r']
    mean_colors = ['darkred', 'darkblue', 'darkgrey', 'darkgreen',
                   'darkorange']
    # plot in reverse order so reds are final plot and always on top
    for nrun, run in reversed(list(enumerate(run_list))):
        try:
            cache = cache_in + '_' + str(nrun)
        except TypeError:
            cache = None
        # add bs distribution plots
        cbar = plot_bs_dists(run, fthetas, axes[:len(fthetas)],
                             parallel=parallel,
                             ftheta_lims=ftheta_lims, cache=cache,
                             n_simulate=n_simulate, nx=nx, ny=ny,
                             rasterize_contours=rasterize_contours,
                             mean_color=mean_colors[nrun],
                             colormap=colormaps[nrun],
                             tqdm_kwargs=tqdm_kwargs)
        # add colorbar
        colorbar_plot = plt.colorbar(cbar, cax=axes[len(fthetas) + nrun],
                                     ticks=[1, 2, 3])
        colorbar_plot.solids.set_edgecolor('face')
        colorbar_plot.ax.set_yticklabels([])
        if nrun == len(run_list) - 1:
            colorbar_plot.ax.set_yticklabels(
                [r'$1\sigma$', r'$2\sigma$', r'$3\sigma$'])
    # Format axis ticks and labels
    for nax, ax in enumerate(axes[:len(fthetas)]):
        ax.set_yticks([])
        ax.set_xlabel(labels[nax])
        if ax.is_first_col():
            ax.set_ylabel('probability')
        # Prune final xtick label so it doesn't overlap with next plot
        prune = 'upper' if nax != len(fthetas) - 1 else None
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(
            nbins=5, prune=prune))
    np.random.set_state(state)  # return to original random state
    return fig