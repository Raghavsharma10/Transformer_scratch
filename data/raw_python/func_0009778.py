def make_sine_surface(dims=DEFAULT_DIMS, offset=0.5, scale=1.0):
    """Makes a surface from the 3D sine function.

    Args:
        dims (pair): the dimensions of the surface to create
        offset (float): an offset applied to the function
        scale (float): a scale applied to the sine frequency

    Returns:
        surface: A surface.
    """
    gradients = (np.array(make_gradients(dims)) - offset) * scale * np.pi
    return np.sin(np.linalg.norm(gradients, axis=0))