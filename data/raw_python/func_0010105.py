def from_yiq(y, i, q, alpha=1.0, wref=_DEFAULT_WREF):
    """Create a new instance based on the specifed YIQ values.

    Parameters:
      :y:
        The Y component value [0...1]
      :i:
        The I component value [0...1]
      :q:
        The Q component value [0...1]
      :alpha:
        The color transparency [0...1], default is opaque
      :wref:
        The whitepoint reference, default is 2° D65.

    Returns:
      A grapefruit.Color instance.

    >>> Color.from_yiq(0.5922, 0.45885,-0.05)
    Color(0.999902, 0.499955, -6.7e-05, 1.0)
    >>> Color.from_yiq(0.5922, 0.45885,-0.05, 0.5)
    Color(0.999902, 0.499955, -6.7e-05, 0.5)

    """
    return Color(yiq_to_rgb(y, i, q), 'rgb', alpha, wref)