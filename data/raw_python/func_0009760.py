def show_plot(plot, width=PREVIEW_WIDTH, height=PREVIEW_HEIGHT):
    """Preview a plot in a jupyter notebook.

    Args:
        plot (list): the plot to display (list of layers)
        width (int): the width of the preview
        height (int): the height of the preview
    
    Returns:
        An object that renders in Jupyter as the provided plot
    """
    return SVG(data=plot_to_svg(plot, width, height))