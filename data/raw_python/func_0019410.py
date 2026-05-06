def shot_chart_jointplot(x, y, data=None, kind="scatter", title="", color="b",
                         cmap=None, xlim=(-250, 250), ylim=(422.5, -47.5),
                         court_color="gray", court_lw=1, outer_lines=False,
                         flip_court=False, size=(12, 11), space=0,
                         despine=False, joint_kws=None, marginal_kws=None,
                         **kwargs):
    """
    Returns a seaborn JointGrid using sns.jointplot

    Parameters
    ----------

    x, y : strings or vector
        The x and y coordinates of the shots taken. They can be passed in as
        vectors (such as a pandas Series) or as column names from the pandas
        DataFrame passed into ``data``.
    data : DataFrame, optional
        DataFrame containing shots where ``x`` and ``y`` represent the
        shot location coordinates.
    kind : { "scatter", "kde", "hex" }, optional
        The kind of shot chart to create.
    title : str, optional
        The title for the plot.
    color : matplotlib color, optional
        Color used to plot the shots
    cmap : matplotlib Colormap object or name, optional
        Colormap for the range of data values. If one isn't provided, the
        colormap is derived from the valuue passed to ``color``. Used for KDE
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
        If ``True`` orients the hoop towards the bottom of the plot.  Default
        is ``False``, which orients the court where the hoop is towards the top
        of the plot.
    gridsize : int, optional
        Number of hexagons in the x-direction.  The default is calculated using
        the Freedman-Diaconis method.
    size : tuple, optional
        The width and height of the plot in inches.
    space : numeric, optional
        The space between the joint and marginal plots.
    {joint, marginal}_kws : dicts
        Additional kewyord arguments for joint and marginal plot components.
    kwargs : key, value pairs
        Keyword arguments for matplotlib Collection properties or seaborn plots.

    Returns
    -------
     grid : JointGrid
        The JointGrid object with the shot chart plotted on it.

   """

    # If a colormap is not provided, then it is based off of the color
    if cmap is None:
        cmap = sns.light_palette(color, as_cmap=True)

    if kind not in ["scatter", "kde", "hex"]:
        raise ValueError("kind must be 'scatter', 'kde', or 'hex'.")

    grid = sns.jointplot(x=x, y=y, data=data, stat_func=None, kind=kind,
                         space=0, color=color, cmap=cmap, joint_kws=joint_kws,
                         marginal_kws=marginal_kws, **kwargs)

    grid.fig.set_size_inches(size)

    # A joint plot has 3 Axes, the first one called ax_joint
    # is the one we want to draw our court onto and adjust some other settings
    ax = grid.ax_joint

    if not flip_court:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    else:
        ax.set_xlim(xlim[::-1])
        ax.set_ylim(ylim[::-1])

    draw_court(ax, color=court_color, lw=court_lw, outer_lines=outer_lines)

    # Get rid of axis labels and tick marks
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.tick_params(labelbottom='off', labelleft='off')

    # Add a title
    ax.set_title(title, y=1.2, fontsize=18)

    # Set the spines to match the rest of court lines, makes outer_lines
    # somewhate unnecessary
    for spine in ax.spines:
        ax.spines[spine].set_lw(court_lw)
        ax.spines[spine].set_color(court_color)
        # set the margin joint spines to be same as the rest of the plot
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