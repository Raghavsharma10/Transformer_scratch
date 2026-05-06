def get_path_style(path, fill=True):
    """Get the style dictionary for matplotlib path objects"""
    style = {}
    style['alpha'] = path.get_alpha()
    if style['alpha'] is None:
        style['alpha'] = 1
    style['edgecolor'] = color_to_hex(path.get_edgecolor())
    if fill:
        style['facecolor'] = color_to_hex(path.get_facecolor())
    else:
        style['facecolor'] = 'none'
    style['edgewidth'] = path.get_linewidth()
    style['dasharray'] = get_dasharray(path)
    style['zorder'] = path.get_zorder()
    return style