def make_bubble_surface(dims=DEFAULT_DIMS, repeat=3):
    """Makes a surface from the product of sine functions on each axis.

    Args:
        dims (pair): the dimensions of the surface to create
        repeat (int): the frequency of the waves is set to ensure this many
            repetitions of the function
    
    Returns:
        surface: A surface.
    """
    gradients = make_gradients(dims)
    return (
        np.sin((gradients[0] - 0.5) * repeat * np.pi) *
        np.sin((gradients[1] - 0.5) * repeat * np.pi))