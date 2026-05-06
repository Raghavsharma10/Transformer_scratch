def rgb_to_ints(r, g=None, b=None):
  """Convert the color in the standard [0...1] range to ints in the [0..255] range.

  Parameters:
    :r:
      The Red component value [0...1]
    :g:
      The Green component value [0...1]
    :b:
      The Blue component value [0...1]

  Returns:
    The color as an (r, g, b) tuple in the range:
    r[0...255],
    g[0...2551],
    b[0...2551]

  >>> rgb_to_ints(1, 0.5, 0)
  (255, 128, 0)

  """
  if type(r) in [list,tuple]:
    r, g, b = r
  return tuple(int(round(v*255)) for v in (r, g, b))