def make_analogous_scheme(self, angle=30, mode='ryb'):
    """Return two colors analogous to this one.

    Args:
      :angle:
        The angle between the hues of the created colors and this one.
      :mode:
        Select which color wheel to use for the generation (ryb/rgb).

    Returns:
      A tuple of grapefruit.Colors analogous to this one.

    >>> c1 = Color.from_hsl(30, 1, 0.5)

    >>> c2, c3 = c1.make_analogous_scheme(angle=60, mode='rgb')
    >>> c2.hsl
    (330.0, 1.0, 0.5)
    >>> c3.hsl
    (90.0, 1.0, 0.5)

    >>> c2, c3 = c1.make_analogous_scheme(angle=10, mode='rgb')
    >>> c2.hsl
    (20.0, 1.0, 0.5)
    >>> c3.hsl
    (40.0, 1.0, 0.5)

    """
    h, s, l = self.__hsl

    if mode == 'ryb': h = rgb_to_ryb(h)
    h += 360
    h1 = (h - angle) % 360
    h2 = (h + angle) % 360
    if mode == 'ryb':
      h1 = ryb_to_rgb(h1)
      h2 = ryb_to_rgb(h2)

    return (Color((h1, s,  l), 'hsl', self.__a, self.__wref),
        Color((h2, s,  l), 'hsl', self.__a, self.__wref))