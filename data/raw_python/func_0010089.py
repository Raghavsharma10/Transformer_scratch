def rgb_to_cmy(r, g=None, b=None):
  """Convert the color from RGB coordinates to CMY.

  Parameters:
    :r:
      The Red component value [0...1]
    :g:
      The Green component value [0...1]
    :b:
      The Blue component value [0...1]

  Returns:
    The color as an (c, m, y) tuple in the range:
    c[0...1],
    m[0...1],
    y[0...1]

  >>> rgb_to_cmy(1, 0.5, 0)
  (0, 0.5, 1)

  """
  if type(r) in [list,tuple]:
    r, g, b = r
  return (1-r, 1-g, 1-b)