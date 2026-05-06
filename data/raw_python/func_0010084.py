def xyz_to_rgb(x, y=None, z=None):
  """Convert the color from CIE XYZ coordinates to sRGB.

  .. note::

     Compensation for sRGB gamma correction is applied before converting.

  Parameters:
    :x:
      The X component value [0...1]
    :y:
      The Y component value [0...1]
    :z:
      The Z component value [0...1]

  Returns:
    The color as an (r, g, b) tuple in the range:
    r[0...1],
    g[0...1],
    b[0...1]

  >>> '(%g, %g, %g)' % xyz_to_rgb(0.488941, 0.365682, 0.0448137)
  '(1, 0.5, 6.81883e-08)'

  """
  if type(x) in [list,tuple]:
    x, y, z = x
  r =  (x * 3.2406255) - (y * 1.5372080) - (z * 0.4986286)
  g = -(x * 0.9689307) + (y * 1.8757561) + (z * 0.0415175)
  b =  (x * 0.0557101) - (y * 0.2040211) + (z * 1.0569959)
  return tuple((((v <= _srgbGammaCorrInv) and [v * 12.92] or [(1.055 * (v ** (1/2.4))) - 0.055])[0] for v in (r, g, b)))