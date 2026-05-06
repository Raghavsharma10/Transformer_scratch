def rgb_to_pil(r, g=None, b=None):
  """Convert the color from RGB to a PIL-compatible integer.

  Parameters:
    :r:
      The Red component value [0...1]
    :g:
      The Green component value [0...1]
    :b:
      The Blue component value [0...1]

  Returns:
    A PIL compatible integer (0xBBGGRR).

  >>> '0x%06x' % rgb_to_pil(1, 0.5, 0)
  '0x0080ff'

  """
  if type(r) in [list,tuple]:
    r, g, b = r
  r, g, b = [min(int(round(v*255)), 255) for v in (r, g, b)]
  return (b << 16) + (g << 8) + r