def with_white_ref(self, wref, labAsRef=False):
    """Create a new instance based on this one with a new white reference.

    Parameters:
      :wref:
        The whitepoint reference.
      :labAsRef:
        If True, the L*a*b* values of the current instance are used as reference
        for the new color; otherwise, the RGB values are used as reference.

    Returns:
      A grapefruit.Color instance.


    >>> c = Color.from_rgb(1.0, 0.5, 0.0, 1.0, WHITE_REFERENCE['std_D65'])

    >>> c2 = c.with_white_ref(WHITE_REFERENCE['sup_D50'])
    >>> c2.rgb
    (1.0, 0.5, 0.0)
    >>> '(%g, %g, %g)' % c2.white_ref
    '(0.967206, 1, 0.81428)'

    >>> c2 = c.with_white_ref(WHITE_REFERENCE['sup_D50'], labAsRef=True)
    >>> '(%g, %g, %g)' % c2.rgb
    '(1.01463, 0.490341, -0.148133)'
    >>> '(%g, %g, %g)' % c2.white_ref
    '(0.967206, 1, 0.81428)'
    >>> '(%g, %g, %g)' % c.lab
    '(66.9518, 0.430841, 0.739692)'
    >>> '(%g, %g, %g)' % c2.lab
    '(66.9518, 0.430841, 0.739693)'

    """
    if labAsRef:
      l, a, b = self.lab
      return Color.from_lab(l, a, b, self.__a, wref)
    else:
      return Color(self.__rgb, 'rgb', self.__a, wref)