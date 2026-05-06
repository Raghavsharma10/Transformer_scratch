def from_yuv(y, u, v, alpha=1.0, wref=_DEFAULT_WREF):
    """Create a new instance based on the specifed YUV values.

    Parameters:
      :y:
        The Y component value [0...1]
      :u:
        The U component value [-0.436...0.436]
      :v:
        The V component value [-0.615...0.615]
      :alpha:
        The color transparency [0...1], default is opaque
      :wref:
        The whitepoint reference, default is 2° D65.

    Returns:
      A grapefruit.Color instance.

    >>> Color.from_yuv(0.5925, -0.2916, 0.3575)
    Color(0.999989, 0.500015, -6.3e-05, 1.0)
    >>> Color.from_yuv(0.5925, -0.2916, 0.3575, 0.5)
    Color(0.999989, 0.500015, -6.3e-05, 0.5)

    """
    return Color(yuv_to_rgb(y, u, v), 'rgb', alpha, wref)