def websafe_dither(self):
    """Return the two websafe colors nearest to this one.

    Returns:
      A tuple of two grapefruit.Color instances which are the two
      web safe colors closest this one.

    >>> c = Color.from_rgb(1.0, 0.45, 0.0)
    >>> c1, c2 = c.websafe_dither()
    >>> c1
    Color(1.0, 0.4, 0.0, 1.0)
    >>> c2
    Color(1.0, 0.6, 0.0, 1.0)

    """
    return (
      Color(rgb_to_websafe(*self.__rgb), 'rgb', self.__a, self.__wref),
      Color(rgb_to_websafe(alt=True, *self.__rgb), 'rgb', self.__a, self.__wref))