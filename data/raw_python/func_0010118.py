def alpha_blend(self, other):
    """Alpha-blend this color on the other one.

    Args:
      :other:
        The grapefruit.Color to alpha-blend with this one.

    Returns:
      A grapefruit.Color instance which is the result of alpha-blending
      this color on the other one.

    >>> c1 = Color.from_rgb(1, 0.5, 0, 0.2)
    >>> c2 = Color.from_rgb(1, 1, 1, 0.8)
    >>> c3 = c1.alpha_blend(c2)
    >>> c3
    Color(1.0, 0.875, 0.75, 0.84)

    """
    # get final alpha channel
    fa = self.__a + other.__a - (self.__a * other.__a)

    # get percentage of source alpha compared to final alpha
    if fa==0: sa = 0
    else: sa = min(1.0, self.__a/other.__a)

    # destination percentage is just the additive inverse
    da = 1.0 - sa

    sr, sg, sb = [v * sa for v in self.__rgb]
    dr, dg, db = [v * da for v in other.__rgb]

    return Color((sr+dr, sg+dg, sb+db), 'rgb', fa, self.__wref)