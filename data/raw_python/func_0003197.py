def grid_search(grid_scores, change, subset=None, kind='line', cmap=None,
                ax=None):
    """
    Plot results from a sklearn grid search by changing two parameters at most.

    Parameters
    ----------
    grid_scores : list of named tuples
        Results from a sklearn grid search (get them using the
        `grid_scores_` parameter)
    change : str or iterable with len<=2
        Parameter to change
    subset : dictionary-like
        parameter-value(s) pairs to subset from grid_scores.
        (e.g. ``{'n_estimartors': [1, 10]}``), if None all combinations will be
        used.
    kind : ['line', 'bar']
        This only applies whe change is a single parameter. Changes the
        type of plot
    cmap : matplotlib Colormap
        This only applies when change are two parameters. Colormap used for
        the matrix. If None uses a modified version of matplotlib's OrRd
        colormap.
    ax: matplotlib Axes
        Axes object to draw the plot onto, otherwise uses current Axes

    Returns
    -------
    ax: matplotlib Axes
        Axes containing the plot

    Examples
    --------

    .. plot:: ../../examples/grid_search.py

    """
    if change is None:
        raise ValueError(('change can\'t be None, you need to select at least'
                          ' one value to make the plot.'))

    if ax is None:
        ax = plt.gca()

    if cmap is None:
        cmap = default_heatmap()

    if isinstance(change, string_types) or len(change) == 1:
        return _grid_search_single(grid_scores, change, subset, kind, ax)
    elif len(change) == 2:
        return _grid_search_double(grid_scores, change, subset, cmap, ax)
    else:
        raise ValueError('change must have length 1 or 2 or be a string')