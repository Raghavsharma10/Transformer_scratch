def lab_to_xyz(l, a=None, b=None, wref=_DEFAULT_WREF):
  """Convert the color from CIE L*a*b* to CIE 1931 XYZ.

  Parameters:
    :l:
      The L component [0...100]
    :a:
      The a component [-1...1]
    :b:
      The a component [-1...1]
    :wref:
      The whitepoint reference, default is 2° D65.

  Returns:
    The color as an (x, y, z) tuple in the range:
    x[0...q],
    y[0...1],
    z[0...1]

  >>> '(%g, %g, %g)' % lab_to_xyz(66.9518, 0.43084, 0.739692)
  '(0.48894, 0.365682, 0.0448137)'

  >>> '(%g, %g, %g)' % lab_to_xyz(66.9518, 0.411663, 0.67282, WHITE_REFERENCE['std_D50'])
  '(0.488942, 0.365682, 0.0448137)'

  """
  if type(l) in [list,tuple]:
    l, a, b = l
  y = (l + 16) / 116
  x = (a / 5.0) + y
  z = y - (b / 2.0)
  return tuple((((v > 0.206893) and [v**3] or [(v - _sixteenHundredsixteenth) / 7.787])[0] * w for v, w in zip((x, y, z), wref)))