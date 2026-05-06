def rgb_to_yiq(r, g=None, b=None):
  """Convert the color from RGB to YIQ.

  Parameters:
    :r:
      The Red component value [0...1]
    :g:
      The Green component value [0...1]
    :b:
      The Blue component value [0...1]

  Returns:
    The color as an (y, i, q) tuple in the range:
    y[0...1],
    i[0...1],
    q[0...1]

  >>> '(%g, %g, %g)' % rgb_to_yiq(1, 0.5, 0)
  '(0.592263, 0.458874, -0.0499818)'

  """
  if type(r) in [list,tuple]:
    r, g, b = r

  y = (r * 0.29895808) + (g * 0.58660979) + (b *0.11443213)
  i = (r * 0.59590296) - (g * 0.27405705) - (b *0.32184591)
  q = (r * 0.21133576) - (g * 0.52263517) + (b *0.31129940)
  return (y, i, q)