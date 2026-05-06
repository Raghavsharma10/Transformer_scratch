def from_xyz(x, y, z, alpha=1.0, wref=_DEFAULT_WREF):
    """Create a new instance based on the specifed CIE-XYZ values.

    Parameters:
      :x:
        The Red component value [0...1]
      :y:
        The Green component value [0...1]
      :z:
        The Blue component value [0...1]
      :alpha:
        The color transparency [0...1], default is opaque
      :wref:
        The whitepoint reference, default is 2° D65.

    Returns:
      A grapefruit.Color instance.

    >>> Color.from_xyz(0.488941, 0.365682, 0.0448137)
    Color(1.0, 0.5, 0.0, 1.0)
    >>> Color.from_xyz(0.488941, 0.365682, 0.0448137, 0.5)
    Color(1.0, 0.5, 0.0, 0.5)

    """
    return Color(xyz_to_rgb(x, y, z), 'rgb', alpha, wref)