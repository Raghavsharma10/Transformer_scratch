def complementary_color(self, mode='ryb'):
    """Create a new instance which is the complementary color of this one.

    Parameters:
      :mode:
        Select which color wheel to use for the generation (ryb/rgb).


    Returns:
      A grapefruit.Color instance.

    >>> Color.from_hsl(30, 1, 0.5).complementary_color(mode='rgb')
    Color(0.0, 0.5, 1.0, 1.0)
    >>> Color.from_hsl(30, 1, 0.5).complementary_color(mode='rgb').hsl
    (210.0, 1.0, 0.5)

    """
    h, s, l = self.__hsl

    if mode == 'ryb': h = rgb_to_ryb(h)
    h = (h+180)%360
    if mode == 'ryb': h = ryb_to_rgb(h)

    return Color((h, s, l), 'hsl', self.__a, self.__wref)