def get_colormap(name, *args, **kwargs):
    """Obtain a colormap

    Some colormaps can have additional configuration parameters. Refer to
    their corresponding documentation for more information.

    Parameters
    ----------
    name : str | Colormap
        Colormap name. Can also be a Colormap for pass-through.

    Examples
    --------

        >>> get_colormap('autumn')
        >>> get_colormap('single_hue', hue=10)
    """
    if isinstance(name, BaseColormap):
        cmap = name
    else:
        if not isinstance(name, string_types):
            raise TypeError('colormap must be a Colormap or string name')
        if name not in _colormaps:
            raise KeyError('colormap name %s not found' % name)
        cmap = _colormaps[name]

        if inspect.isclass(cmap):
            cmap = cmap(*args, **kwargs)

    return cmap