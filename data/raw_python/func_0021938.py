def vertical_percent(plot, percent=0.1):
    """
    Using the size of the y axis, return a fraction of that size.
    """
    plot_bottom, plot_top = plot.get_ylim()
    return percent * (plot_top - plot_bottom)