def project_surface(surface, angle=DEFAULT_ANGLE):
    """Returns the height of the surface when projected at the given angle.

    Args:
        surface (surface): the surface to project
        angle (float): the angle at which to project the surface

    Returns:
        surface: A projected surface.
    """
    z_coef = np.sin(np.radians(angle))
    y_coef = np.cos(np.radians(angle))

    surface_height, surface_width = surface.shape
    slope = np.tile(np.linspace(0., 1., surface_height), [surface_width, 1]).T

    return slope * y_coef + surface * z_coef