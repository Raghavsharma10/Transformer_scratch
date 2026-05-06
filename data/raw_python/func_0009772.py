def make_lines_texture(num_lines=10, resolution=50):
    """Makes a texture consisting of a given number of horizontal lines.

    Args:
        num_lines (int): the number of lines to draw
        resolution (int): the number of midpoints on each line

    Returns:
        A texture.
    """
    x, y = np.meshgrid(
        np.hstack([np.linspace(0, 1, resolution), np.nan]),
        np.linspace(0, 1, num_lines),
    )
    
    y[np.isnan(x)] = np.nan
    return x.flatten(), y.flatten()