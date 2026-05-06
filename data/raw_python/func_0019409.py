def shot_chart_jointgrid(x, y, data=None, joint_type="scatter", title="",
                         joint_color="b", cmap=None,  xlim=(-250, 250),
                         ylim=(422.5, -47.5), court_color="gray", court_lw=1,
                         outer_lines=False, flip_court=False,
                         joint_kde_shade=True, gridsize=None,
                         marginals_color="b", marginals_type="both",
                         marginals_kde_shade=True, size=(12, 11), space=0,
                         despine=False, joint_kws=None, marginal_kws=None,
                         **kwargs):

    """
    Returns a JointGrid object containing the shot chart.

    This function allows for more flexibility in customizing your shot chart
    than the ``shot_chart_jointplot`` function.

    Parameters
    ----------

    x, y : strings or vector
        The x and y coordinates of the shots taken. They can be passed in as
        vectors (such as a pandas Series) or as columns from the pandas
        DataFrame passed into ``data``.
    data : DataFrame, optional
        DataFrame containing shots where ``x`` and ``y`` represent the shot
        location coordinates.
    joint_type : { "scatter", "kde", "hex" }, optional
        The type of shot chart for the joint plot.
    title : str, optional
        The title for the plot.
    joint_color : matplotlib color, optional
        Color used to plot the shots on the joint plot.
    cmap : matplotlib Colormap object or name, optional
        Colormap for the range of data values. If one isn't provided, the
        colormap is derived from the value passed to ``color``. Used for KDE
        and Hexbin joint plots.
    {x, y}lim : two-tuples, optional
        The axis limits of the plot.  The defaults represent the out of bounds
        lines and half court line.
    court_color : matplotlib color, optional
        The color of the court lines.
    court_lw : float, optional
        The linewidth the of the court lines.
    outer_lines : boolean, optional
        If ``True`` the out of bound lines are drawn in as a matplotlib
        Rectangle.
    flip_court : boolean, optional
        If ``True`` orients the hoop towards the bottom of the plot. Default is
        ``False``, which orients the court where the hoop is towards the top of
        the plot.
    joint_kde_shade : boolean, optional
        Default is ``True``, which shades in the KDE contours on the joint plot.
    gridsize : int, optional
        Number of hexagons in the x-direction. The default is calculated using
        the Freedman-Diaconis method.
    marginals_color : matplotlib color, optional
        Color used to plot the shots on the marginal plots.
    marginals_type : { "both", "hist", "kde"}, optional
        The type of plot for the marginal plots.
    marginals_kde_shade : boolean, optional
        Default is ``True``, which shades in the KDE contours on the marginal
        plots.
    size : tuple, optional
        The width and height of the plot in inches.
    space : numeric, optional
        The space between the joint and marginal plots.
    despine : boolean, optional
        If ``True``, removes the spines.
    {joint, marginal}_kws : dicts
        Additional kewyord arguments for joint and marginal plot components.
    kwargs : key, value pairs
        Keyword arguments for matplotlib Collection properties or seaborn plots.

    Returns
    -------
     grid : JointGrid
        The JointGrid object with the shot chart plotted on it.

    """

    # The joint_kws and marginal_kws idea was taken from seaborn
    # Create the default empty kwargs for joint and marginal plots
    if joint_kws is None:
        joint_kws = {}
    joint_kws.update(kwargs)

    if marginal_kws is None:
        marginal_kws = {}

    # If a colormap is not provided, then it is based off of the joint_color
    if cmap is None:
        cmap = sns.light_palette(joint_color, as_cmap=True)

    # Flip the court so that the hoop is by the bottom of the plot
    if flip_court:
        xlim = xlim[::-1]
        ylim = ylim[::-1]

    # Create the JointGrid to draw the shot chart plots onto
    grid = sns.JointGrid(x=x, y=y, data=data, xlim=xlim, ylim=ylim,
                         space=space)

    # Joint Plot
    # Create the main plot of the joint shot chart
    if joint_type == "scatter":
        grid = grid.plot_joint(plt.scatter, color=joint_color, **joint_kws)

    elif joint_type == "kde":
        grid = grid.plot_joint(sns.kdeplot, cmap=cmap,
                               shade=joint_kde_shade, **joint_kws)

    elif joint_type == "hex":
        if gridsize is None:
            # Get the number of bins for hexbin using Freedman-Diaconis rule
            # This is idea was taken from seaborn, which got the calculation
            # from http://stats.stackexchange.com/questions/798/
            from seaborn.distributions import _freedman_diaconis_bins
            x_bin = _freedman_diaconis_bins(x)
            y_bin = _freedman_diaconis_bins(y)
            gridsize = int(np.mean([x_bin, y_bin]))

        grid = grid.plot_joint(plt.hexbin, gridsize=gridsize, cmap=cmap,
                               **joint_kws)

    else:
        raise ValueError("joint_type must be 'scatter', 'kde', or 'hex'.")

    # Marginal plots
    # Create the plots on the axis of the main plot of the joint shot chart.
    if marginals_type == "both":
        grid = grid.plot_marginals(sns.distplot, color=marginals_color,
                                   **marginal_kws)

    elif marginals_type == "hist":
        grid = grid.plot_marginals(sns.distplot, color=marginals_color,
                                   kde=False, **marginal_kws)

    elif marginals_type == "kde":
        grid = grid.plot_marginals(sns.kdeplot, color=marginals_color,
                                   shade=marginals_kde_shade, **marginal_kws)

    else:
        raise ValueError("marginals_type must be 'both', 'hist', or 'kde'.")

    # Set the size of the joint shot chart
    grid.fig.set_size_inches(size)

    # Extract the the first axes, which is the main plot of the
    # joint shot chart, and draw the court onto it
    ax = grid.fig.get_axes()[0]
    draw_court(ax, color=court_color, lw=court_lw, outer_lines=outer_lines)

    # Get rid of the axis labels
    grid.set_axis_labels(xlabel="", ylabel="")
    # Get rid of all tick labels
    ax.tick_params(labelbottom="off", labelleft="off")
    # Set the title above the top marginal plot
    ax.set_title(title, y=1.2, fontsize=18)

    # Set the spines to match the rest of court lines, makes outer_lines
    # somewhate unnecessary
    for spine in ax.spines:
        ax.spines[spine].set_lw(court_lw)
        ax.spines[spine].set_color(court_color)
        # set the marginal spines to be the same as the rest of the spines
        grid.ax_marg_x.spines[spine].set_lw(court_lw)
        grid.ax_marg_x.spines[spine].set_color(court_color)
        grid.ax_marg_y.spines[spine].set_lw(court_lw)
        grid.ax_marg_y.spines[spine].set_color(court_color)

    if despine:
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

    return grid