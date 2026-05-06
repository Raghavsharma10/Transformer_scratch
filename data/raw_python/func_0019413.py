def bokeh_shot_chart(data, x="LOC_X", y="LOC_Y", fill_color="#1f77b4",
                     scatter_size=10, fill_alpha=0.4, line_alpha=0.4,
                     court_line_color='gray', court_line_width=1,
                     hover_tool=False, tooltips=None, **kwargs):

    # TODO: Settings for hover tooltip
    """
    Returns a figure with both FGA and basketball court lines drawn onto it.

    This function expects data to be a ColumnDataSource with the x and y values
    named "LOC_X" and "LOC_Y".  Otherwise specify x and y.

    Parameters
    ----------

    data : DataFrame
        The DataFrame that contains the shot chart data.
    x, y : str, optional
        The x and y coordinates of the shots taken.
    fill_color : str, optional
        The fill color of the shots. Can be a a Hex value.
    scatter_size : int, optional
        The size of the dots for the scatter plot.
    fill_alpha : float, optional
        Alpha value for the shots. Must be a floating point value between 0
        (transparent) to 1 (opaque).
    line_alpha : float, optiona
        Alpha value for the outer lines of the plotted shots. Must be a
        floating point value between 0 (transparent) to 1 (opaque).
    court_line_color : str, optional
        The color of the court lines. Can be a a Hex value.
    court_line_width : float, optional
        The linewidth the of the court lines in pixels.
    hover_tool : boolean, optional
        If ``True``, creates hover tooltip for the plot.
    tooltips : List of tuples, optional
        Provides the information for the the hover tooltip.

    Returns
    -------
    fig : Figure
        The Figure object with the shot chart plotted on it.

    """
    source = ColumnDataSource(data)

    fig = figure(width=700, height=658, x_range=[-250, 250],
                 y_range=[422.5, -47.5], min_border=0, x_axis_type=None,
                 y_axis_type=None, outline_line_color="black", **kwargs)

    fig.scatter(x, y, source=source, size=scatter_size, color=fill_color,
                alpha=fill_alpha, line_alpha=line_alpha)

    bokeh_draw_court(fig, line_color=court_line_color,
                     line_width=court_line_width)

    if hover_tool:
        hover = HoverTool(renderers=[fig.renderers[0]], tooltips=tooltips)
        fig.add_tools(hover)

    return fig