def rgb_to_xyz(r, g=None, b=None):
  """Convert the color from sRGB to CIE XYZ.

  The methods assumes that the RGB coordinates are given in the sRGB
  colorspace (D65).

  .. note::

     Compensation for the sRGB gamma correction is applied before converting.

  Parameters:
    :r:
      The Red component value [0...1]
    :g:
      The Green component value [0...1]
    :b:
      The Blue component value [0...1]

  Returns:
    The color as an (x, y, z) tuple in the range:
    x[0...1],
    y[0...1],
    z[0...1]

  >>> '(%g, %g, %g)' % rgb_to_xyz(1, 0.5, 0)
  '(0.488941, 0.365682, 0.0448137)'

  """
  if type(r) in [list,tuple]:
    r, g, b = r

  r, g, b = [((v <= 0.03928) and [v / 12.92] or [((v+0.055) / 1.055) **2.4])[0] for v in (r, g, b)]

  x = (r * 0.4124) + (g * 0.3576) + (b * 0.1805)
  y = (r * 0.2126) + (g * 0.7152) + (b * 0.0722)
  z = (r * 0.0193) + (g * 0.1192) + (b * 0.9505)
  return (x, y, z)