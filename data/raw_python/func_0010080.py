def yiq_to_rgb(y, i=None, q=None):
  """Convert the color from YIQ coordinates to RGB.

  Parameters:
    :y:
      Tte Y component value [0...1]
    :i:
      The I component value [0...1]
    :q:
      The Q component value [0...1]

  Returns:
    The color as an (r, g, b) tuple in the range:
    r[0...1],
    g[0...1],
    b[0...1]

  >>> '({}, {}, {})'.format(*[round(v, 6) for v in yiq_to_rgb(0.592263, 0.458874, -0.0499818)])
  '(1.0, 0.5, 1e-06)'

  """
  if type(y) in [list,tuple]:
    y, i, q = y
  r = y + (i * 0.9562) + (q * 0.6210)
  g = y - (i * 0.2717) - (q * 0.6485)
  b = y - (i * 1.1053) + (q * 1.7020)
  return (r, g, b)