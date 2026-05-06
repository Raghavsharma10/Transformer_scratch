def make_spiral_texture(spirals=6.0, ccw=False, offset=0.0, resolution=1000):
    """Makes a texture consisting of a spiral from the origin.

    Args:
        spirals (float): the number of rotations to make
        ccw (bool): make spirals counter-clockwise (default is clockwise)
        offset (float): if non-zero, spirals start offset by this amount
        resolution (int): number of midpoints along the spiral

    Returns:
        A texture.
    """
    dist = np.sqrt(np.linspace(0., 1., resolution))
    if ccw:
        direction = 1.
    else:
        direction = -1.
    angle = dist * spirals * np.pi * 2. * direction
    spiral_texture = (
        (np.cos(angle) * dist / 2.) + 0.5,
        (np.sin(angle) * dist / 2.) + 0.5
    )
    return spiral_texture