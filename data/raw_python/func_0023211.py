def get_layout(name, *args, **kwargs):
    """
    Retrieve a graph layout

    Some graph layouts accept extra options. Please refer to their
    documentation for more information.

    Parameters
    ----------
    name : string
        The name of the layout. The variable `AVAILABLE_LAYOUTS`
        contains all available layouts.
    *args
        Positional arguments which are passed to the layout.
    **kwargs
        Keyword arguments which are passed to the layout.

    Returns
    -------
    layout : callable
        The callable generator which will calculate the graph layout
    """
    if name not in _layout_map:
        raise KeyError("Graph layout '%s' not found. Should be one of %s"
                       % (name, AVAILABLE_LAYOUTS))

    layout = _layout_map[name]

    if inspect.isclass(layout):
        layout = layout(*args, **kwargs)

    return layout