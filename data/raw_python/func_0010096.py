def pil_to_rgb(pil):
  """Convert the color from a PIL-compatible integer to RGB.

  Parameters:
    pil: a PIL compatible color representation (0xBBGGRR)
  Returns:
    The color as an (r, g, b) tuple in the range:
    the range:
    r: [0...1]
    g: [0...1]
    b: [0...1]

  >>> '(%g, %g, %g)' % pil_to_rgb(0x0080ff)
  '(1, 0.501961, 0)'

  """
  r = 0xff & pil
  g = 0xff & (pil >> 8)
  b = 0xff & (pil >> 16)
  return tuple((v / 255.0 for v in (r, g, b)))