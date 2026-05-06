def hsv_to_rgb(h, s=None, v=None):
  """Convert the color from RGB coordinates to HSV.

  Parameters:
    :h:
      The Hus component value [0...1]
    :s:
      The Saturation component value [0...1]
    :v:
      The Value component [0...1]

  Returns:
    The color as an (r, g, b) tuple in the range:
    r[0...1],
    g[0...1],
    b[0...1]

  >>> hsv_to_rgb(30.0, 1.0, 0.5)
  (0.5, 0.25, 0.0)

  """
  if type(h) in [list,tuple]:
    h, s, v = h

  if s==0: return (v, v, v)   # achromatic (gray)

  h /= 60.0
  h = h % 6.0

  i = int(h)
  f = h - i
  if not(i&1): f = 1-f     # if i is even

  m = v * (1.0 - s)
  n = v * (1.0 - (s * f))

  if i==0: return (v, n, m)
  if i==1: return (n, v, m)
  if i==2: return (m, v, n)
  if i==3: return (m, n, v)
  if i==4: return (n, m, v)
  return (v, m, n)