def from_hsv(h, s, v, alpha=1.0, wref=_DEFAULT_WREF):
    """Create a new instance based on the specifed HSV values.

    Parameters:
      :h:
        The Hus component value [0...1]
      :s:
        The Saturation component value [0...1]
      :v:
        The Value component [0...1]
      :alpha:
        The color transparency [0...1], default is opaque
      :wref:
        The whitepoint reference, default is 2° D65.

    Returns:
      A grapefruit.Color instance.

    >>> Color.from_hsv(30, 1, 1)
    Color(1.0, 0.5, 0.0, 1.0)
    >>> Color.from_hsv(30, 1, 1, 0.5)
    Color(1.0, 0.5, 0.0, 0.5)

    """
    h2, s, l = rgb_to_hsl(*hsv_to_rgb(h, s, v))
    return Color((h, s, l), 'hsl', alpha, wref)