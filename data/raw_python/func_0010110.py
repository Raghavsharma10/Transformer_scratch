def from_cmyk(c, m, y, k, alpha=1.0, wref=_DEFAULT_WREF):
    """Create a new instance based on the specifed CMYK values.

    Parameters:
      :c:
        The Cyan component value [0...1]
      :m:
        The Magenta component value [0...1]
      :y:
        The Yellow component value [0...1]
      :k:
        The Black component value [0...1]
      :alpha:
        The color transparency [0...1], default is opaque
      :wref:
        The whitepoint reference, default is 2° D65.

    Returns:
      A grapefruit.Color instance.

    >>> Color.from_cmyk(1, 0.32, 0, 0.5)
    Color(0.0, 0.34, 0.5, 1.0)
    >>> Color.from_cmyk(1, 0.32, 0, 0.5, 0.5)
    Color(0.0, 0.34, 0.5, 0.5)

    """
    return Color(cmy_to_rgb(*cmyk_to_cmy(c, m, y, k)), 'rgb', alpha, wref)