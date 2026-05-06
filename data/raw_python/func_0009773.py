def make_grid_texture(num_h_lines=10, num_v_lines=10, resolution=50):
    """Makes a texture consisting of a grid of vertical and horizontal lines.

    Args:
        num_h_lines (int): the number of horizontal lines to draw
        num_v_lines (int): the number of vertical lines to draw
        resolution (int): the number of midpoints to draw on each line

    Returns:
        A texture.
    """
    x_h, y_h = make_lines_texture(num_h_lines, resolution)
    y_v, x_v = make_lines_texture(num_v_lines, resolution)
    return np.concatenate([x_h, x_v]), np.concatenate([y_h, y_v])