def rgb_to_hsv(r, g=None, b=None):
  """Convert the color from RGB coordinates to HSV.

  Parameters:
    :r:
      The Red component value [0...1]
    :g:
      The Green component value [0...1]
    :b:
      The Blue component value [0...1]

  Returns:
    The color as an (h, s, v) tuple in the range:
    h[0...360],
    s[0...1],
    v[0...1]

  >>> rgb_to_hsv(1, 0.5, 0)
  (30.0, 1.0, 1.0)

  """
  if type(r) in [list,tuple]:
    r, g, b = r

  v = float(max(r, g, b))
  d = v - min(r, g, b)
  if d==0: return (0.0, 0.0, v)
  s = d / v

  dr, dg, db = [(v - val) / d for val in (r, g, b)]

  if r==v:
    h = db - dg             # between yellow & magenta
  elif g==v:
    h = 2.0 + dr - db       # between cyan & yellow
  else: # b==v
    h = 4.0 + dg - dr       # between magenta & cyan

  h = (h*60.0) % 360.0
  return (h, s, v)