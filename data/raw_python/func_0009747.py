def fit_texture(layer):
    """Fits a layer into a texture by scaling each axis to (0, 1).

    Does not preserve aspect ratio (TODO: make this an option).

    Args:
        layer (layer): the layer to scale

    Returns:
        texture: A texture.
    """
    x, y = layer
    x = (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x))
    y = (y - np.nanmin(y)) / (np.nanmax(y) - np.nanmin(y))
    return x, y