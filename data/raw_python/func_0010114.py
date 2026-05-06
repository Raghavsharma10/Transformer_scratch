def desaturate(self, level):
    """Create a new instance based on this one but less saturated.

    Parameters:
      :level:
        The amount by which the color should be desaturated to produce
        the new one [0...1].

    Returns:
      A grapefruit.Color instance.

    >>> Color.from_hsl(30, 0.5, 0.5).desaturate(0.25)
    Color(0.625, 0.5, 0.375, 1.0)
    >>> Color.from_hsl(30, 0.5, 0.5).desaturate(0.25).hsl
    (30.0, 0.25, 0.5)

    """
    h, s, l = self.__hsl
    return Color((h, max(s - level, 0), l), 'hsl', self.__a, self.__wref)