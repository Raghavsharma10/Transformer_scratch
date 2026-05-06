def rgb2short(r, g, b):
    """
    Converts RGB values to the nearest equivalent xterm-256 color.
    """
    # Using list of snap points, convert RGB value to cube indexes
    r, g, b = [len(tuple(s for s in snaps if s < x)) for x in (r, g, b)]

    # Simple colorcube transform
    return (r * 36) + (g * 6) + b + 16