def rgb_to_hsl(r, g=None, b=None):
  """Convert the color from RGB coordinates to HSL.

  Parameters:
    :r:
      The Red component value [0...1]
    :g:
      The Green component value [0...1]
    :b:
      The Blue component value [0...1]

  Returns:
    The color as an (h, s, l) tuple in the range:
    h[0...360],
    s[0...1],
    l[0...1]

  >>> rgb_to_hsl(1, 0.5, 0)
  (30.0, 1.0, 0.5)

  """
  if type(r) in [list,tuple]:
    r, g, b = r

  minVal = min(r, g, b)       # min RGB value
  maxVal = max(r, g, b)       # max RGB value

  l = (maxVal + minVal) / 2.0
  if minVal==maxVal:
    return (0.0, 0.0, l)    # achromatic (gray)

  d = maxVal - minVal         # delta RGB value

  if l < 0.5: s = d / (maxVal + minVal)
  else: s = d / (2.0 - maxVal - minVal)

  dr, dg, db = [(maxVal-val) / d for val in (r, g, b)]

  if r==maxVal:
    h = db - dg
  elif g==maxVal:
    h = 2.0 + dr - db
  else:
    h = 4.0 + dg - dr

  h = (h*60.0) % 360.0
  return (h, s, l)