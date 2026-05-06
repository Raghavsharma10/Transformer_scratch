def map_texture_to_surface(texture, surface):
    """Returns values on a surface for points on a texture.

    Args:
        texture (texture): the texture to trace over the surface
        surface (surface): the surface to trace along

    Returns:
        an array of surface heights for each point in the
        texture. Line separators (i.e. values that are ``nan`` in
        the texture) will be ``nan`` in the output, so the output
        will have the same dimensions as the x/y axes in the
        input texture.
    """
    texture_x, texture_y = texture
    surface_h, surface_w = surface.shape

    surface_x = np.clip(
        np.int32(surface_w * texture_x - 1e-9), 0, surface_w - 1)
    surface_y = np.clip(
        np.int32(surface_h * texture_y - 1e-9), 0, surface_h - 1)

    surface_z = surface[surface_y, surface_x]
    return surface_z