def xyz_to_lab(x, y=None, z=None, wref=_DEFAULT_WREF):
  """Convert the color from CIE XYZ to CIE L*a*b*.

  Parameters:
    :x:
      The X component value [0...1]
    :y:
      The Y component value [0...1]
    :z:
      The Z component value [0...1]
    :wref:
      The whitepoint reference, default is 2° D65.

  Returns:
    The color as an (L, a, b) tuple in the range:
    L[0...100],
    a[-1...1],
    b[-1...1]

  >>> '(%g, %g, %g)' % xyz_to_lab(0.488941, 0.365682, 0.0448137)
  '(66.9518, 0.430841, 0.739692)'

  >>> '(%g, %g, %g)' % xyz_to_lab(0.488941, 0.365682, 0.0448137, WHITE_REFERENCE['std_D50'])
  '(66.9518, 0.41166, 0.67282)'

  """
  if type(x) in [list,tuple]:
    x, y, z = x
  # White point correction
  x /= wref[0]
  y /= wref[1]
  z /= wref[2]

  # Nonlinear distortion and linear transformation
  x, y, z = [((v > 0.008856) and [v**_oneThird] or [(7.787 * v) + _sixteenHundredsixteenth])[0] for v in (x, y, z)]

  # Vector scaling
  l = (116 * y) - 16
  a = 5.0 * (x - y)
  b = 2.0 * (y - z)

  return (l, a, b)