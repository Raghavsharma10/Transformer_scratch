def project_texture(texture_xy, texture_z, angle=DEFAULT_ANGLE):
    """Creates a texture by adding z-values to an existing texture and projecting.

    When working with surfaces there are two ways to accomplish the same thing:

    1. project the surface and map a texture to the projected surface
    2. map a texture to the surface, and then project the result

    The first method, which does not use this function, is preferred because
    it is easier to do occlusion removal that way. This function is provided
    for cases where you do not wish to generate a surface (and don't care about
    occlusion removal.)

    Args:
        texture_xy (texture): the texture to project
        texture_z (np.array): the Z-values to use in the projection
        angle (float): the angle to project at, in degrees (0 = overhead, 90 = side view)

    Returns:
        layer: A layer.
    """
    z_coef = np.sin(np.radians(angle))
    y_coef = np.cos(np.radians(angle))
    surface_x, surface_y = texture
    return (surface_x, -surface_y * y_coef + surface_z * z_coef)