def yuv_to_rgb(y, u=None, v=None):
  """Convert the color from YUV coordinates to RGB.

  Parameters:
    :y:
      The Y component value [0...1]
    :u:
      The U component value [-0.436...0.436]
    :v:
      The V component value [-0.615...0.615]

  Returns:
    The color as an (r, g, b) tuple in the range:
    r[0...1],
    g[0...1],
    b[0...1]

  >>> '(%g, %g, %g)' % yuv_to_rgb(0.5925, -0.2916, 0.3575)
  '(0.999989, 0.500015, -6.3276e-05)'

  """
  if type(y) in [list,tuple]:
    y, u, v = y
  r = y + (v * 1.13983)
  g = y - (u * 0.39465) - (v * 0.58060)
  b = y + (u * 2.03211)
  return (r, g, b)