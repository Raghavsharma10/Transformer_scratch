def _mpl_to_vispy(fig):
    """Convert a given matplotlib figure to vispy

    This function is experimental and subject to change!
    Requires matplotlib and mplexporter.

    Parameters
    ----------
    fig : instance of matplotlib Figure
        The populated figure to display.

    Returns
    -------
    canvas : instance of Canvas
        The resulting vispy Canvas.
    """
    renderer = VispyRenderer()
    exporter = Exporter(renderer)
    with warnings.catch_warnings(record=True):  # py3k mpl warning
        exporter.run(fig)
    renderer._vispy_done()
    return renderer.canvas