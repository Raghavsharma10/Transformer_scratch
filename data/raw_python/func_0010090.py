def cmy_to_rgb(c, m=None, y=None):
  """Convert the color from CMY coordinates to RGB.

  Parameters:
    :c:
      The Cyan component value [0...1]
    :m:
      The Magenta component value [0...1]
    :y:
      The Yellow component value [0...1]

  Returns:
    The color as an (r, g, b) tuple in the range:
    r[0...1],
    g[0...1],
    b[0...1]

  >>> cmy_to_rgb(0, 0.5, 1)
  (1, 0.5, 0)

  """
  if type(c) in [list,tuple]:
    c, m, y = c
  return (1-c, 1-m, 1-y)