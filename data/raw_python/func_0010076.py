def hsl_to_rgb(h, s=None, l=None):
  """Convert the color from HSL coordinates to RGB.

  Parameters:
    :h:
      The Hue component value [0...1]
    :s:
      The Saturation component value [0...1]
    :l:
      The Lightness component value [0...1]

  Returns:
    The color as an (r, g, b) tuple in the range:
    r[0...1],
    g[0...1],
    b[0...1]

  >>> hsl_to_rgb(30.0, 1.0, 0.5)
  (1.0, 0.5, 0.0)

  """
  if type(h) in [list,tuple]:
    h, s, l = h

  if s==0: return (l, l, l)   # achromatic (gray)

  if l<0.5: n2 = l * (1.0 + s)
  else: n2 = l+s - (l*s)

  n1 = (2.0 * l) - n2

  h /= 60.0
  hueToRgb = _hue_to_rgb
  r = hueToRgb(n1, n2, h + 2)
  g = hueToRgb(n1, n2, h)
  b = hueToRgb(n1, n2, h - 2)

  return (r, g, b)