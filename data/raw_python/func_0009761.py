def calculate_view_box(layers, aspect_ratio, margin=DEFAULT_VIEW_BOX_MARGIN):
    """Calculates the size of the SVG viewBox to use.

    Args:
        layers (list): the layers in the image
        aspect_ratio (float): the height of the output divided by the width
        margin (float): minimum amount of buffer to add around the image, relative
            to the total dimensions

    Returns:
        tuple: a 4-tuple of floats representing the viewBox according to SVG
            specifications ``(x, y, width, height)``.
    """
    min_x = min(np.nanmin(x) for x, y in layers)
    max_x = max(np.nanmax(x) for x, y in layers)
    min_y = min(np.nanmin(y) for x, y in layers)
    max_y = max(np.nanmax(y) for x, y in layers)
    height = max_y - min_y
    width = max_x - min_x

    if height > width * aspect_ratio:
        adj_height = height * (1. + margin)
        adj_width = adj_height / aspect_ratio
    else:
        adj_width = width * (1. + margin)
        adj_height = adj_width * aspect_ratio

    width_buffer = (adj_width - width) / 2.
    height_buffer = (adj_height - height) / 2.

    return (
        min_x - width_buffer,
        min_y - height_buffer,
        adj_width,
        adj_height
    )