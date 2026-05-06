def blend(self, other, percent=0.5):
    """blend this color with the other one.

    Args:
      :other:
        the grapefruit.Color to blend with this one.

    Returns:
      A grapefruit.Color instance which is the result of blending
      this color on the other one.

    >>> c1 = Color.from_rgb(1, 0.5, 0, 0.2)
    >>> c2 = Color.from_rgb(1, 1, 1, 0.6)
    >>> c3 = c1.blend(c2)
    >>> c3
    Color(1.0, 0.75, 0.5, 0.4)

    """
    dest = 1.0 - percent
    rgb = tuple(((u * percent) + (v * dest) for u, v in zip(self.__rgb, other.__rgb)))
    a = (self.__a * percent) + (other.__a * dest)
    return Color(rgb, 'rgb', a, self.__wref)