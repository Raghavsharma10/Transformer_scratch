def project_texture_on_surface(texture, surface, angle=DEFAULT_ANGLE):
    """Maps a texture onto a surface, then projects to 2D and returns a layer.

    Args:
        texture (texture): the texture to project
        surface (surface): the surface to project onto
        angle (float): the projection angle in degrees (0 = top-down, 90 = side view)

    Returns:
        layer: A layer.
    """
    projected_surface = project_surface(surface, angle)
    texture_x, _ = texture
    texture_y = map_texture_to_surface(texture, projected_surface)
    return texture_x, texture_y