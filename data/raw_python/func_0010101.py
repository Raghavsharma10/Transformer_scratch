def ryb_to_rgb(hue):
  """Maps a hue on Itten's RYB color wheel to the standard RGB wheel.

  Parameters:
    :hue:
      The hue on Itten's RYB color wheel [0...360]

  Returns:
    An approximation of the corresponding hue on the standard RGB wheel.

  >>> ryb_to_rgb(15)
  8.0

  """
  d = hue % 15
  i = int(hue / 15)
  x0 = _RgbWheel[i]
  x1 = _RgbWheel[i+1]
  return x0 + (x1-x0) * d / 15