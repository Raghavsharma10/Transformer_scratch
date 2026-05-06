def make_gradients(dims=DEFAULT_DIMS):
    """Makes a pair of gradients to generate textures from numpy primitives.

    Args:
        dims (pair): the dimensions of the surface to create

    Returns:
        pair: A pair of surfaces.
    """
    return np.meshgrid(
        np.linspace(0.0, 1.0, dims[0]),
        np.linspace(0.0, 1.0, dims[1])
    )