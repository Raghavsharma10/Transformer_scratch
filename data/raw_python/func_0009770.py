def _remove_hidden_parts(projected_surface):
    """Removes parts of a projected surface that are not visible.

    Args:
        projected_surface (surface): the surface to use

    Returns:
        surface: A projected surface.
    """
    surface = np.copy(projected_surface)
    surface[~_make_occlusion_mask(projected_surface)] = np.nan
    return surface