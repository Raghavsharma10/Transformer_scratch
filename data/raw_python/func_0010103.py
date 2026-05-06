def from_hsl(h, s, l, alpha=1.0, wref=_DEFAULT_WREF):
    """Create a new instance based on the specifed HSL values.

    Parameters:
      :h:
        The Hue component value [0...1]
      :s:
        The Saturation component value [0...1]
      :l:
        The Lightness component value [0...1]
      :alpha:
        The color transparency [0...1], default is opaque
      :wref:
        The whitepoint reference, default is 2° D65.

    Returns:
      A grapefruit.Color instance.

    >>> Color.from_hsl(30, 1, 0.5)
    Color(1.0, 0.5, 0.0, 1.0)
    >>> Color.from_hsl(30, 1, 0.5, 0.5)
    Color(1.0, 0.5, 0.0, 0.5)

    """
    return Color((h, s, l), 'hsl', alpha, wref)