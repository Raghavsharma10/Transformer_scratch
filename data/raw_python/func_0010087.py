def cmyk_to_cmy(c, m=None, y=None, k=None):
  """Convert the color from CMYK coordinates to CMY.

  Parameters:
    :c:
      The Cyan component value [0...1]
    :m:
      The Magenta component value [0...1]
    :y:
      The Yellow component value [0...1]
    :k:
      The Black component value [0...1]

  Returns:
    The color as an (c, m, y) tuple in the range:
    c[0...1],
    m[0...1],
    y[0...1]

  >>> '(%g, %g, %g)' % cmyk_to_cmy(1, 0.32, 0, 0.5)
  '(1, 0.66, 0.5)'

  """
  if type(c) in [list,tuple]:
    c, m, y, k = c
  mk = 1-k
  return ((c*mk + k), (m*mk + k), (y*mk + k))