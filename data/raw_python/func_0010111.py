def from_html(html, alpha=1.0, wref=_DEFAULT_WREF):
    """Create a new instance based on the specifed HTML color definition.

    Parameters:
      :html:
        The HTML definition of the color (#RRGGBB or #RGB or a color name).
      :alpha:
        The color transparency [0...1], default is opaque.
      :wref:
        The whitepoint reference, default is 2° D65.

    Returns:
      A grapefruit.Color instance.

    >>> Color.from_html('#ff8000')
    Color(1.0, 0.501961, 0.0, 1.0)
    >>> Color.from_html('ff8000')
    Color(1.0, 0.501961, 0.0, 1.0)
    >>> Color.from_html('#f60')
    Color(1.0, 0.4, 0.0, 1.0)
    >>> Color.from_html('f60')
    Color(1.0, 0.4, 0.0, 1.0)
    >>> Color.from_html('lemonchiffon')
    Color(1.0, 0.980392, 0.803922, 1.0)
    >>> Color.from_html('#ff8000', 0.5)
    Color(1.0, 0.501961, 0.0, 0.5)

    """
    return Color(html_to_rgb(html), 'rgb', alpha, wref)