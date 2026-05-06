def ints_to_rgb(r, g=None, b=None):
  """Convert ints in the [0...255] range to the standard [0...1] range.

  Parameters:
    :r:
      The Red component value [0...255]
    :g:
      The Green component value [0...255]
    :b:
      The Blue component value [0...255]

  Returns:
    The color as an (r, g, b) tuple in the range:
    r[0...1],
    g[0...1],
    b[0...1]

  >>> '(%g, %g, %g)' % ints_to_rgb((255, 128, 0))
  '(1, 0.501961, 0)'

  """
  if type(r) in [list,tuple]:
    r, g, b = r
  return tuple(float(v) / 255.0 for v in [r, g, b])