def cmy_to_cmyk(c, m=None, y=None):
  """Convert the color from CMY coordinates to CMYK.

  Parameters:
    :c:
      The Cyan component value [0...1]
    :m:
      The Magenta component value [0...1]
    :y:
      The Yellow component value [0...1]

  Returns:
    The color as an (c, m, y, k) tuple in the range:
    c[0...1],
    m[0...1],
    y[0...1],
    k[0...1]

  >>> '(%g, %g, %g, %g)' % cmy_to_cmyk(1, 0.66, 0.5)
  '(1, 0.32, 0, 0.5)'

  """
  if type(c) in [list,tuple]:
    c, m, y = c
  k = min(c, m, y)
  if k==1.0: return (0.0, 0.0, 0.0, 1.0)
  mk = 1.0-k
  return ((c-k) / mk, (m-k) / mk, (y-k) / mk, k)