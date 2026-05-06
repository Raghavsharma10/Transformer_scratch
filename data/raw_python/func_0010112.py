def from_pil(pil, alpha=1.0, wref=_DEFAULT_WREF):
    """Create a new instance based on the specifed PIL color.

    Parameters:
      :pil:
        A PIL compatible color representation (0xBBGGRR)
      :alpha:
        The color transparency [0...1], default is opaque
      :wref:
        The whitepoint reference, default is 2° D65.

    Returns:
      A grapefruit.Color instance.

    >>> Color.from_pil(0x0080ff)
    Color(1.0, 0.501961, 0.0, 1.0)
    >>> Color.from_pil(0x0080ff, 0.5)
    Color(1.0, 0.501961, 0.0, 0.5)

    """
    return Color(pil_to_rgb(pil), 'rgb', alpha, wref)