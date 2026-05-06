def make_noise_surface(dims=DEFAULT_DIMS, blur=10, seed=None):
    """Makes a surface by generating random noise and blurring it.

    Args:
        dims (pair): the dimensions of the surface to create
        blur (float): the amount of Gaussian blur to apply
        seed (int): a random seed to use (optional)
    
    Returns:
        surface: A surface.
    """
    if seed is not None:
        np.random.seed(seed)

    return gaussian_filter(np.random.normal(size=dims), blur)