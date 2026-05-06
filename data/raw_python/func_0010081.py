def rgb_to_yuv(r, g=None, b=None):
  """Convert the color from RGB coordinates to YUV.

  Parameters:
    :r:
      The Red component value [0...1]
    :g:
      The Green component value [0...1]
    :b:
      The Blue component value [0...1]

  Returns:
    The color as an (y, u, v) tuple in the range:
    y[0...1],
    u[-0.436...0.436],
    v[-0.615...0.615]

  >>> '(%g, %g, %g)' % rgb_to_yuv(1, 0.5, 0)
  '(0.5925, -0.29156, 0.357505)'

  """
  if type(r) in [list,tuple]:
    r, g, b = r

  y =  (r * 0.29900) + (g * 0.58700) + (b * 0.11400)
  u = -(r * 0.14713) - (g * 0.28886) + (b * 0.43600)
  v =  (r * 0.61500) - (g * 0.51499) - (b * 0.10001)
  return (y, u, v)