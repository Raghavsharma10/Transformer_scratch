def get_axis_properties(axis):
    """Return the property dictionary for a matplotlib.Axis instance"""
    props = {}
    label1On = axis._major_tick_kw.get('label1On', True)

    if isinstance(axis, matplotlib.axis.XAxis):
        if label1On:
            props['position'] = "bottom"
        else:
            props['position'] = "top"
    elif isinstance(axis, matplotlib.axis.YAxis):
        if label1On:
            props['position'] = "left"
        else:
            props['position'] = "right"
    else:
        raise ValueError("{0} should be an Axis instance".format(axis))

    # Use tick values if appropriate
    locator = axis.get_major_locator()
    props['nticks'] = len(locator())
    if isinstance(locator, ticker.FixedLocator):
        props['tickvalues'] = list(locator())
    else:
        props['tickvalues'] = None

    # Find tick formats
    formatter = axis.get_major_formatter()
    if isinstance(formatter, ticker.NullFormatter):
        props['tickformat'] = ""
    elif isinstance(formatter, ticker.FixedFormatter):
        props['tickformat'] = list(formatter.seq)
    elif not any(label.get_visible() for label in axis.get_ticklabels()):
        props['tickformat'] = ""
    else:
        props['tickformat'] = None

    # Get axis scale
    props['scale'] = axis.get_scale()

    # Get major tick label size (assumes that's all we really care about!)
    labels = axis.get_ticklabels()
    if labels:
        props['fontsize'] = labels[0].get_fontsize()
    else:
        props['fontsize'] = None

    # Get associated grid
    props['grid'] = get_grid_style(axis)

    return props