def rgb_to_websafe(r, g=None, b=None, alt=False):
  """Convert the color from RGB to 'web safe' RGB

  Parameters:
    :r:
      The Red component value [0...1]
    :g:
      The Green component value [0...1]
    :b:
      The Blue component value [0...1]
    :alt:
      If True, use the alternative color instead of the nearest one.
      Can be used for dithering.

  Returns:
    The color as an (r, g, b) tuple in the range:
    the range:
    r[0...1],
    g[0...1],
    b[0...1]

  >>> '(%g, %g, %g)' % rgb_to_websafe(1, 0.55, 0.0)
  '(1, 0.6, 0)'

  """
  if type(r) in [list,tuple]:
    r, g, b = r
  websafeComponent = _websafe_component
  return tuple((websafeComponent(v, alt) for v in (r, g, b)))