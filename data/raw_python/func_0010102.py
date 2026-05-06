def from_rgb(r, g, b, alpha=1.0, wref=_DEFAULT_WREF):
    """Create a new instance based on the specifed RGB values.

    Parameters:
      :r:
        The Red component value [0...1]
      :g:
        The Green component value [0...1]
      :b:
        The Blue component value [0...1]
      :alpha:
        The color transparency [0...1], default is opaque
      :wref:
        The whitepoint reference, default is 2° D65.

    Returns:
      A grapefruit.Color instance.

    >>> Color.from_rgb(1.0, 0.5, 0.0)
    Color(1.0, 0.5, 0.0, 1.0)
    >>> Color.from_rgb(1.0, 0.5, 0.0, 0.5)
    Color(1.0, 0.5, 0.0, 0.5)

    """
    return Color((r, g, b), 'rgb', alpha, wref)