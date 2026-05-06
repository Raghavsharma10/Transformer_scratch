def get_line_style(line):
    """Get the style dictionary for matplotlib line objects"""
    style = {}
    style['alpha'] = line.get_alpha()
    if style['alpha'] is None:
        style['alpha'] = 1
    style['color'] = color_to_hex(line.get_color())
    style['linewidth'] = line.get_linewidth()
    style['dasharray'] = get_dasharray(line)
    style['zorder'] = line.get_zorder()
    return style