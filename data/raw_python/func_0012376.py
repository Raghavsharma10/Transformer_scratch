def kde_plot_df(df, xlims=None, **kwargs):
    """Plots kde estimates of distributions of samples in each cell of the
    input pandas DataFrame.

    There is one subplot for each dataframe column, and on each subplot there
    is one kde line.

    Parameters
    ----------
    df: pandas data frame
        Each cell must contain a 1d numpy array of samples.
    xlims: dict, optional
        Dictionary of xlimits - keys are column names and values are lists of
        length 2.
    num_xticks: int, optional
        Number of xticks on each subplot.
    figsize: tuple, optional
        Size of figure in inches.
    nrows: int, optional
        Number of rows of subplots.
    ncols: int, optional
        Number of columns of subplots.
    normalize: bool, optional
        If true, kde plots are normalized to have the same area under their
        curves. If False, their max value is set to 1.
    legend: bool, optional
        Should a legend be added?
    legend_kwargs: dict, optional
        Additional kwargs for legend.

    Returns
    -------
    fig: matplotlib figure
    """
    assert xlims is None or isinstance(xlims, dict)
    figsize = kwargs.pop('figsize', (6.4, 1.5))
    num_xticks = kwargs.pop('num_xticks', None)
    nrows = kwargs.pop('nrows', 1)
    ncols = kwargs.pop('ncols', int(np.ceil(len(df.columns) / nrows)))
    normalize = kwargs.pop('normalize', True)
    legend = kwargs.pop('legend', False)
    legend_kwargs = kwargs.pop('legend_kwargs', {})
    if kwargs:
        raise TypeError('Unexpected **kwargs: {0}'.format(kwargs))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    for nax, col in enumerate(df):
        if nrows == 1:
            ax = axes[nax]
        else:
            ax = axes[nax // ncols, nax % ncols]
        supmin = df[col].apply(np.min).min()
        supmax = df[col].apply(np.max).max()
        support = np.linspace(supmin - 0.1 * (supmax - supmin),
                              supmax + 0.1 * (supmax - supmin), 200)
        handles = []
        labels = []
        for name, samps in df[col].iteritems():
            pdf = scipy.stats.gaussian_kde(samps)(support)
            if not normalize:
                pdf /= pdf.max()
            handles.append(ax.plot(support, pdf, label=name)[0])
            labels.append(name)
        ax.set_ylim(bottom=0)
        ax.set_yticks([])
        if xlims is not None:
            try:
                ax.set_xlim(xlims[col])
            except KeyError:
                pass
        ax.set_xlabel(col)
        if num_xticks is not None:
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(
                nbins=num_xticks))
    if legend:
        fig.legend(handles, labels, **legend_kwargs)
    return fig