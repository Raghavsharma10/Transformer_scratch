def color_to_hex(color):
    """Convert matplotlib color code to hex color code"""
    if color is None or colorConverter.to_rgba(color)[3] == 0:
        return 'none'
    else:
        rgb = colorConverter.to_rgb(color)
        return '#{0:02X}{1:02X}{2:02X}'.format(*(int(255 * c) for c in rgb))