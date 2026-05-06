def rgb_to_greyscale(r, g=None, b=None):
  """Convert the color from RGB to its greyscale equivalent

  Parameters:
    :r:
      The Red component value [0...1]
    :g:
      The Green component value [0...1]
    :b:
      The Blue component value [0...1]

  Returns:
    The color as an (r, g, b) tuple in the range:
    the range:
    r[0...1],
    g[0...1],
    b[0...1]

  >>> '(%g, %g, %g)' % rgb_to_greyscale(1, 0.8, 0)
  '(0.6, 0.6, 0.6)'

  """
  if type(r) in [list,tuple]:
    r, g, b = r
  v = (r + g + b) / 3.0
  return (v, v, v)