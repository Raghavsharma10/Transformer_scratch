def write_plot(plot, filename, width=DEFAULT_PAGE_WIDTH, height=DEFAULT_PAGE_HEIGHT, unit=DEFAULT_PAGE_UNIT):
    """Writes a plot SVG to a file.

    Args:
        plot (list): a list of layers to plot
        filename (str): the name of the file to write
        width (float): the width of the output SVG
        height (float): the height of the output SVG
        unit (str): the unit of the height and width
    """
    svg = plot_to_svg(plot, width, height, unit)
    with open(filename, 'w') as outfile:
        outfile.write(svg)