def project_and_occlude_texture(texture, surface, angle=DEFAULT_ANGLE):
    """Projects a texture onto a surface with occluded areas removed.

    Args:
        texture (texture): the texture to map to the projected surface
        surface (surface): the surface to project
        angle (float): the angle to project at, in degrees (0 = overhead, 90 = side view)

    Returns:
        layer: A layer.
    """
    projected_surface = project_surface(surface, angle)
    projected_surface = _remove_hidden_parts(projected_surface)
    texture_y = map_texture_to_surface(texture, projected_surface)
    texture_x, _ = texture
    return texture_x, texture_y