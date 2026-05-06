def rgb_to_ryb(hue):
  """Maps a hue on the RGB color wheel to Itten's RYB wheel.

  Parameters:
    :hue:
      The hue on the RGB color wheel [0...360]

  Returns:
    An approximation of the corresponding hue on Itten's RYB wheel.

  >>> rgb_to_ryb(15)
  26.0

  """
  d = hue % 15
  i = int(hue / 15)
  x0 = _RybWheel[i]
  x1 = _RybWheel[i+1]
  return x0 + (x1-x0) * d / 15